#!/usr/bin/env python3
import sys, os, json, argparse, pickle, ahocorasick

CACHE_FILENAME = "glitch_tokens.ahc"

def load_tokens(path):
    data = json.load(open(path, encoding="utf‑8"))
    return data.get("glitch_tokens", [])

def build_and_cache(tokens, cache_path):
    A = ahocorasick.Automaton()
    for tok in tokens:
        A.add_word(tok, tok)
    A.make_automaton()
    with open(cache_path, "wb") as f:
        pickle.dump(tokens, f)
        pickle.dump(A, f)
    return tokens, A

def load_cache(cache_path):
    with open(cache_path, "rb") as f:
        tokens = pickle.load(f)
        A = pickle.load(f)
    return tokens, A

def get_automaton(tokens, json_path):
    if os.path.exists(CACHE_FILENAME):
        try:
            cached_tokens, A = load_cache(CACHE_FILENAME)
            if cached_tokens == tokens:
                return tokens, A
        except Exception:
            pass
    return build_and_cache(tokens, CACHE_FILENAME)

def scan_exact_lines(tokens_set, infile):
    for line in infile:
        s = line.rstrip("\n")
        if s in tokens_set:
            print(s)

def scan_substring(A, infile):
    for line in infile:
        line = line.rstrip("\n")
        for _, tok in A.iter(line):
            print(line)
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="JSON file path")
    parser.add_argument("-i", "--input", default=None, help="input file or stdin")
    parser.add_argument("--substring", action="store_true",
                        help="match tokens inside lines instead of exact-line")
    args = parser.parse_args()

    tokens = load_tokens(args.json) or []
    tokens, A = get_automaton(tokens, args.json)
    tokens_set = set(tokens)

    inf = open(args.input, encoding="utf‑8") if args.input else sys.stdin
    if args.substring:
        scan_substring(A, inf)
    else:
        scan_exact_lines(tokens_set, inf)

if __name__ == "__main__":
    main()
