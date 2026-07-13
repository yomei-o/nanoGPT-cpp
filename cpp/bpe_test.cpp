// Verify the BPE tokenizer against known GPT-2 token ids (ASCII English).
#include "bpe.h"
#include <cstdio>
#include <vector>
#include <string>
using namespace gpt;

static bool check(const BPETokenizer& t, const std::string& s, std::vector<int> expect) {
    auto got = t.encode(s);
    bool ok = (got == expect);
    std::printf("  encode(\"%s\") = [", s.c_str());
    for (size_t i = 0; i < got.size(); i++) std::printf("%s%d", i ? ", " : "", got[i]);
    std::printf("]  %s\n", ok ? "OK" : "MISMATCH");
    if (!ok) {
        std::printf("    expected [");
        for (size_t i = 0; i < expect.size(); i++) std::printf("%s%d", i ? ", " : "", expect[i]);
        std::printf("]\n");
    }
    // round-trip
    std::string back = t.decode(got);
    if (back != s) { std::printf("    round-trip FAIL: \"%s\"\n", back.c_str()); ok = false; }
    return ok;
}

int main() {
    BPETokenizer t;
    if (!t.load("encoder.json", "vocab.bpe")) { std::fprintf(stderr, "load failed\n"); return 1; }
    std::printf("loaded %zu tokens, %zu merges\n", t.encoder.size(), t.bpe_ranks.size());
    bool ok = true;
    ok &= check(t, "Hello world", {15496, 995});
    ok &= check(t, "Hello, I'm a language model,", {15496, 11, 314, 1101, 257, 3303, 2746, 11});
    ok &= check(t, "The quick brown fox jumps over the lazy dog.",
                {464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13});
    ok &= check(t, "GPT-2 is a transformer.", {38, 11571, 12, 17, 318, 257, 47385, 13});
    std::printf("%s\n", ok ? "ALL BPE CHECKS PASSED" : "SOME BPE CHECKS FAILED");
    return ok ? 0 : 1;
}
