// ---------------------------------------------------------------------------
//  bpe.h  --  GPT-2 byte-level BPE tokenizer (for the gpt2 inference driver).
//
//  Loads OpenAI's encoder.json (token string -> id) and vocab.bpe (merge
//  rules), and implements byte-level BPE encode/decode.
//
//  NOTE on fidelity: GPT-2's pre-tokenization uses a Unicode-property regex
//  (\p{L}, \p{N}). Reproducing that exactly in dependency-free C++ is
//  impractical, so this uses an ASCII-faithful splitter: it matches GPT-2's
//  tokenization exactly for ASCII English text (letters/digits/punct/space and
//  the 's/'t/'re/'ve/'m/'ll/'d contractions), and only diverges on non-ASCII
//  letters. The byte-level BPE itself is exact.
// ---------------------------------------------------------------------------
#ifndef NANOGPT_BPE_H
#define NANOGPT_BPE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cctype>
#include <cstdint>

namespace gpt {

struct BPETokenizer {
    std::unordered_map<std::string, int> encoder;   // token string -> id
    std::vector<std::string> decoder;                // id -> token string
    std::unordered_map<std::string, int> bpe_ranks;  // "a b" -> rank
    std::string byte_to_uni[256];                    // byte -> utf8 of mapped codepoint
    std::unordered_map<std::string, int> uni_to_byte;// utf8 -> byte

    static std::string cp_to_utf8(int cp) {
        std::string s;
        if (cp < 0x80) s += (char)cp;
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        return s;
    }

    void build_byte_maps() {
        std::vector<int> bs, cs;
        auto add_range = [&](int a, int b) { for (int c = a; c <= b; c++) { bs.push_back(c); cs.push_back(c); } };
        add_range('!', '~'); add_range(0xA1, 0xAC); add_range(0xAE, 0xFF);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            bool found = false;
            for (int x : bs) if (x == b) { found = true; break; }
            if (!found) { bs.push_back(b); cs.push_back(256 + n); n++; }
        }
        for (size_t i = 0; i < bs.size(); i++) {
            std::string u = cp_to_utf8(cs[i]);
            byte_to_uni[bs[i]] = u;
            uni_to_byte[u] = bs[i];
        }
    }

    // minimal JSON parser for a flat {"string": int, ...} object
    static std::string parse_json_string(const std::string& s, size_t& i) {
        std::string out; // assumes s[i]=='"'
        i++;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') break;
            if (c == '\\') {
                char e = s[i++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case '/': out += '/'; break;
                    case '\\': out += '\\'; break;
                    case '"': out += '"'; break;
                    case 'u': {
                        int cp = std::stoi(s.substr(i, 4), nullptr, 16); i += 4;
                        out += cp_to_utf8(cp);
                        break;
                    }
                    default: out += e;
                }
            } else out += c;
        }
        return out;
    }

    bool load(const std::string& encoder_json, const std::string& vocab_bpe) {
        build_byte_maps();
        // encoder.json
        std::ifstream ef(encoder_json, std::ios::binary);
        if (!ef) return false;
        std::string js((std::istreambuf_iterator<char>(ef)), std::istreambuf_iterator<char>());
        size_t i = 0;
        while (i < js.size() && js[i] != '{') i++;
        i++;
        while (i < js.size()) {
            while (i < js.size() && js[i] != '"' && js[i] != '}') i++;
            if (i >= js.size() || js[i] == '}') break;
            std::string key = parse_json_string(js, i);
            while (i < js.size() && js[i] != ':') i++; i++;
            while (i < js.size() && (js[i] == ' ' || js[i] == '\n')) i++;
            int val = 0; bool neg = false;
            if (js[i] == '-') { neg = true; i++; }
            while (i < js.size() && std::isdigit((unsigned char)js[i])) { val = val * 10 + (js[i] - '0'); i++; }
            if (neg) val = -val;
            encoder[key] = val;
            while (i < js.size() && js[i] != ',' && js[i] != '}') i++;
            if (i < js.size() && js[i] == ',') i++;
            else break;
        }
        decoder.assign(encoder.size(), "");
        for (auto& kv : encoder) if (kv.second >= 0 && kv.second < (int)decoder.size()) decoder[kv.second] = kv.first;

        // vocab.bpe
        std::ifstream vf(vocab_bpe, std::ios::binary);
        if (!vf) return false;
        std::string line; int rank = 0; bool first = true;
        while (std::getline(vf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (first) { first = false; if (!line.empty() && line[0] == '#') continue; }
            if (line.empty()) continue;
            bpe_ranks[line] = rank++;   // "a b" -> rank (already space-separated)
        }
        return !encoder.empty() && !bpe_ranks.empty();
    }

    // ASCII-faithful pre-tokenizer approximating GPT-2's regex
    static std::vector<std::string> pretokenize(const std::string& text) {
        std::vector<std::string> pieces;
        size_t i = 0, n = text.size();
        auto isL = [](unsigned char c) { return std::isalpha(c) != 0; };
        auto isN = [](unsigned char c) { return std::isdigit(c) != 0; };
        auto isS = [](unsigned char c) { return std::isspace(c) != 0; };
        while (i < n) {
            unsigned char c = text[i];
            // contractions: 's 't 're 've 'm 'll 'd
            if (c == '\'' && i + 1 < n) {
                std::string two = text.substr(i, 2), three = text.substr(i, 3);
                static const char* c1[] = {"'s", "'t", "'m", "'d"};
                static const char* c2[] = {"'re", "'ve", "'ll"};
                bool matched = false;
                for (auto p : c2) if (three == p) { pieces.push_back(three); i += 3; matched = true; break; }
                if (matched) continue;
                for (auto p : c1) if (two == p) { pieces.push_back(two); i += 2; matched = true; break; }
                if (matched) continue;
            }
            // optional single leading space
            size_t start = i;
            bool lead_space = (c == ' ');
            size_t j = i + (lead_space ? 1 : 0);
            if (j < n && isL((unsigned char)text[j])) {                 // ' ?\p{L}+
                size_t k = j; while (k < n && isL((unsigned char)text[k])) k++;
                pieces.push_back(text.substr(start, k - start)); i = k; continue;
            }
            if (j < n && isN((unsigned char)text[j])) {                 // ' ?\p{N}+
                size_t k = j; while (k < n && isN((unsigned char)text[k])) k++;
                pieces.push_back(text.substr(start, k - start)); i = k; continue;
            }
            if (j < n && !isS((unsigned char)text[j])) {                // ' ?[^\s\p{L}\p{N}]+
                size_t k = j; while (k < n && !isS((unsigned char)text[k]) && !isL((unsigned char)text[k]) && !isN((unsigned char)text[k])) k++;
                pieces.push_back(text.substr(start, k - start)); i = k; continue;
            }
            // whitespace run. GPT-2 attaches a single trailing *space* (0x20) to
            // the following token as its leading space (the ' ?' in the regex).
            if (isS(c)) {
                size_t k = i; while (k < n && isS((unsigned char)text[k])) k++;
                if (k < n && k - i > 1 && text[k - 1] == ' ') {
                    // >1 whitespace before a non-space, ending in a space: peel that
                    // last space off to lead the next word; emit the rest as a run.
                    pieces.push_back(text.substr(i, k - i - 1));
                    i = k - 1; continue;               // i strictly advances (k-1 > i)
                }
                // otherwise emit the whole whitespace run (this also covers a lone
                // '\n'/'\t'/single space, which must advance to avoid looping).
                pieces.push_back(text.substr(i, k - i)); i = k; continue;
            }
            // fallback
            pieces.push_back(std::string(1, (char)c)); i++;
        }
        return pieces;
    }

    std::vector<std::string> bpe(const std::string& token_uni) const {
        // token_uni is a sequence of utf8 "unicode chars" (each 1-3 bytes)
        std::vector<std::string> word;
        for (size_t i = 0; i < token_uni.size();) {
            int len = 1; unsigned char c = token_uni[i];
            if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
            word.push_back(token_uni.substr(i, len)); i += len;
        }
        if (word.size() <= 1) return word;
        while (true) {
            int best_rank = 1 << 30; size_t best_i = 0; bool found = false;
            for (size_t i = 0; i + 1 < word.size(); i++) {
                auto it = bpe_ranks.find(word[i] + " " + word[i + 1]);
                if (it != bpe_ranks.end() && it->second < best_rank) { best_rank = it->second; best_i = i; found = true; }
            }
            if (!found) break;
            std::vector<std::string> nw;
            for (size_t i = 0; i < word.size();) {
                if (i + 1 < word.size() && i == best_i) { nw.push_back(word[i] + word[i + 1]); i += 2; }
                else { nw.push_back(word[i]); i++; }
            }
            word.swap(nw);
        }
        return word;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        for (const std::string& piece : pretokenize(text)) {
            std::string uni; // map each byte to its unicode-space utf8
            for (unsigned char b : piece) uni += byte_to_uni[b];
            for (const std::string& tok : bpe(uni)) {
                auto it = encoder.find(tok);
                if (it != encoder.end()) ids.push_back(it->second);
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string uni;
        for (int id : ids) if (id >= 0 && id < (int)decoder.size()) uni += decoder[id];
        // map unicode-space utf8 back to raw bytes
        std::string out;
        for (size_t i = 0; i < uni.size();) {
            int len = 1; unsigned char c = uni[i];
            if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
            auto it = uni_to_byte.find(uni.substr(i, len));
            if (it != uni_to_byte.end()) out += (char)it->second;
            i += len;
        }
        return out;
    }
};

} // namespace gpt

#endif // NANOGPT_BPE_H
