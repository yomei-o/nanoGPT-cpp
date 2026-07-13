// ---------------------------------------------------------------------------
//  tokenizer.h  --  character-level tokenizer (matches data/shakespeare_char).
//  (The GPT-2 BPE tokenizer lives in bpe.h, used by the gpt2 inference driver.)
// ---------------------------------------------------------------------------
#ifndef NANOGPT_TOKENIZER_H
#define NANOGPT_TOKENIZER_H

#include <string>
#include <vector>
#include <set>
#include <unordered_map>

namespace gpt {

// Character-level tokenizer: unique characters sorted, mapped to 0..V-1.
struct CharTokenizer {
    std::vector<char> itos;                 // id -> char
    std::unordered_map<char, int> stoi;     // char -> id

    void build_from_text(const std::string& text) {
        std::set<char> uniq(text.begin(), text.end());
        itos.assign(uniq.begin(), uniq.end());   // std::set is sorted
        stoi.clear();
        for (int i = 0; i < (int)itos.size(); i++) stoi[itos[i]] = i;
    }
    void set_vocab(const std::vector<char>& chars) {
        itos = chars; stoi.clear();
        for (int i = 0; i < (int)itos.size(); i++) stoi[itos[i]] = i;
    }
    int vocab_size() const { return (int)itos.size(); }

    std::vector<int> encode(const std::string& s) const {
        std::vector<int> out; out.reserve(s.size());
        for (char c : s) { auto it = stoi.find(c); if (it != stoi.end()) out.push_back(it->second); }
        return out;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string out; out.reserve(ids.size());
        for (int id : ids) if (id >= 0 && id < (int)itos.size()) out.push_back(itos[id]);
        return out;
    }
};

} // namespace gpt

#endif // NANOGPT_TOKENIZER_H
