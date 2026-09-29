// llama.cpp 专家激活直方图（零侵入，不改上游 C++）
// 只用两个公开接口：cb_eval + t->name
//   · cb_eval：llama.h:386 的 ggml_backend_sched_eval_callback，ask=false 时节点已算完并同步
//   · t->name：llama-context.cpp:2483 无条件挂 graph_get_cb，用 ggml_format_name 命名 "<name>-<il>"
// K2 两个 router topk 节点被 cb 的是【完整 argsort 张量】（非 topk view），contiguous I32：
//   ffn_moe_argsort-<il>  (llama-graph.cpp:2057/2060)  ne=[100, n_tok]
//   v_moe_argsort-<il>    (k2-horizon.cpp:329/353)      ne=[64,  n_tok]
// 编译目标：llama-expert-histogram
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

struct hist_state {
    std::mutex mtx;
    std::regex ffn_re;
    std::regex val_re;
    int moe_topk  = 8;
    int val_topk  = 4;
    int n_expert  = 100;
    int n_val_exp = 64;
    long long n_tokens = 0;
    long long n_hits   = 0;   // 累加的 topk 元素总数，用于自检
    std::map<int, std::vector<long long>> ffn;
    std::map<int, std::vector<long long>> value;
    std::vector<std::string> samples;
};

// 读 (i0=c, i1=j) 处的专家编号：argsort 输出列主序，flat = i0 + i1*ne0
static int32_t elem_at(const int32_t * p, int64_t ne0, int c, int64_t j) {
    return p[(size_t) c + (size_t) j * (size_t) ne0];
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * S = (hist_state *) user_data;
    const std::string name(t->name);   // ggml_tensor::name 是 char[GGML_MAX_NAME]，非指针
    const bool is_ffn = std::regex_search(name, S->ffn_re);
    const bool is_val = !is_ffn && std::regex_search(name, S->val_re);

    if (ask) {
        return is_ffn || is_val;   // 仅这两类节点单独停算并触发回调
    }

    if ((is_ffn || is_val) && t->type == GGML_TYPE_I32) {
        const int64_t ne0   = t->ne[0];   // = 该层专家数（直接取自张量，不信 CLI）
        const int64_t n_tok = t->ne[1];
        const int     topk  = is_ffn ? S->moe_topk : S->val_topk;
        if (ne0 < 1 || n_tok < 1 || topk > ne0) return true;

        // 整个 argsort 张量读回主机（完整输出、列主序连续；K2 里 ≈ 100*n_batch*4B）
        const size_t nbytes = ggml_nbytes(t);
        std::vector<int32_t> full(nbytes / sizeof(int32_t));
        if (ggml_backend_buffer_is_host(t->buffer)) {
            std::memcpy(full.data(), t->data, nbytes);
        } else {
            ggml_backend_tensor_get(t, full.data(), 0, nbytes);
        }

        int il = -1;
        { auto pos = name.rfind('-');
          if (pos != std::string::npos) { try { il = std::stoi(name.substr(pos + 1)); } catch (...) {} } }

        std::lock_guard<std::mutex> g(S->mtx);
        auto & m = is_ffn ? S->ffn : S->value;
        auto & cnts = m[il];
        if (cnts.empty()) cnts.assign((size_t) ne0, 0);   // 宽度以张量实测为准
        if (S->samples.size() < 6) S->samples.push_back(name);

        S->n_tokens += n_tok;
        for (int64_t j = 0; j < n_tok; ++j)
            for (int c = 0; c < topk; ++c) {
                const int32_t eid = elem_at(full.data(), ne0, c, j);
                if (eid >= 0 && eid < (int32_t) cnts.size()) { cnts[eid]++; S->n_hits++; }
            }
    }
    return true;
}

static void dump_json(const hist_state & S, const std::string & path, const std::string & model, int n_prompts) {
    std::ofstream o(path);
    auto emit = [&](const char * key, const std::map<int, std::vector<long long>> & m, int k) {
        o << "  \"" << key << "\": {\n";
        for (auto it = m.begin(); it != m.end(); ++it) {
            o << "    \"" << it->first << "\": {\"counts\": [";
            for (size_t i = 0; i < it->second.size(); ++i) { if (i) o << ", "; o << it->second[i]; }
            long long s = 0; for (auto v : it->second) s += v;
            o << "], \"total\": " << s << ", \"expect\": " << (S.n_tokens * k) << "}";
            o << (std::next(it) == m.end() ? "\n" : ",\n");
        }
        o << "  },\n";
    };
    int flo = S.ffn.empty()    ? -1 : S.ffn.begin()->first, fhi = S.ffn.empty()    ? -1 : S.ffn.rbegin()->first;
    int vlo = S.value.empty()  ? -1 : S.value.begin()->first, vhi = S.value.empty() ? -1 : S.value.rbegin()->first;
    o << "{\n";
    o << "  \"meta\": {\"model\": \"" << model << "\", \"n_prompts\": " << n_prompts
      << ", \"token_total\": " << S.n_tokens << ", \"hit_total\": " << S.n_hits
      << ", \"ffn_layers\": [" << flo << ", " << fhi << "]"
      << ", \"value_layers\": [" << vlo << ", " << vhi << "]"
      << ", \"moe_topk\": " << S.moe_topk << ", \"n_expert\": " << S.n_expert
      << ", \"value_topk\": " << S.val_topk << ", \"n_value_expert\": " << S.n_val_exp << "},\n";
    emit("ffn",   S.ffn,   S.moe_topk);
    emit("value", S.value, S.val_topk);
    o << "  \"samples\": [";
    for (size_t i = 0; i < S.samples.size(); ++i) { if (i) o << ", "; o << "\"" << S.samples[i] << "\""; }
    o << "]\n}\n";
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    params.warmup = false;   // 预热不算进直方图

    std::string prompts_file, out_path = "expert_hits.json";
    hist_state S;
    S.ffn_re = std::regex("^ffn_moe_argsort-[0-9]+$");
    S.val_re = std::regex("^v_moe_argsort-[0-9]+$");

    // 手写轻量参数解析，其余全部交给 common_params_parse
    std::vector<char *> fargv;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char * f) { return std::strcmp(argv[i], f) == 0; };
        if ((is("--prompts-file") || is("-pf")) && i + 1 < argc) { prompts_file = argv[++i]; continue; }
        if (is("--out") && i + 1 < argc)                          { out_path   = argv[++i]; continue; }
        if (is("--moe-topk") && i + 1 < argc)                     { S.moe_topk = std::atoi(argv[++i]); continue; }
        if (is("--value-topk") && i + 1 < argc)                   { S.val_topk = std::atoi(argv[++i]); continue; }
        fargv.push_back(argv[i]);
    }
    if (prompts_file.empty()) {
        fprintf(stderr, "error: --prompts-file <path> 必需（每行一条 prompt）\n");
        return 1;
    }
    std::vector<std::string> prompts;
    { std::ifstream f(prompts_file); std::string line;
      while (std::getline(f, line)) { if (!line.empty()) prompts.push_back(line); } }
    if (prompts.empty()) { fprintf(stderr, "error: prompts 文件为空\n"); return 1; }

    std::vector<char *> cargv;
    cargv.push_back(argv[0]);
    for (auto * a : fargv) cargv.push_back(a);
    if (!common_params_parse((int) cargv.size(), cargv.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;

    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &S;

    llama_backend_init();
    auto llama = common_init_from_params(params);
    auto * model = llama->model();
    auto * ctx   = llama->context();
    if (!model || !ctx) { fprintf(stderr, "error: 初始化失败\n"); return 1; }

    for (size_t p = 0; p < prompts.size(); ++p) {
        std::vector<llama_token> toks = common_tokenize(ctx, prompts[p], /*add_special=*/true, /*parse_special=*/true);
        if (toks.empty()) continue;
        fprintf(stderr, "[%zu/%zu] %zu tokens\n", p + 1, prompts.size(), toks.size());
        for (size_t i = 0; i < toks.size();) {
            const int32_t n = std::min<int32_t>((int32_t) (toks.size() - i), params.n_batch);
            auto batch = llama_batch_get_one(toks.data() + i, n);
            if (llama_decode(ctx, batch)) { fprintf(stderr, "decode 失败 @prompt %zu\n", p); break; }
            i += n;
        }
        llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);   // 换 prompt 前清 KV
    }

    dump_json(S, out_path, params.model.path, (int) prompts.size());
    fprintf(stderr, "\n=== 自检 ===\n");
    fprintf(stderr, "token_total   = %lld\n", S.n_tokens);
    fprintf(stderr, "hit_total     = %lld (应 = token_total*%d + token_total*%d，若 prompt 都过 MoVA 层)\n",
            S.n_hits, S.moe_topk, S.val_topk);
    fprintf(stderr, "ffn 层数=%zu  value 层数=%zu\n", S.ffn.size(), S.value.size());
    if (S.samples.empty())
        fprintf(stderr, "!! 未匹配到任何 argsort 节点 —— 检查模型是否 MoE/MoVA，或节点名是否变化\n");
    else
        fprintf(stderr, "节点名样本: %s\n", S.samples[0].c_str());
    fprintf(stderr, "输出: %s\n", out_path.c_str());

    llama_backend_free();
    return 0;
}
