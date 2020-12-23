import tokenizer_cereal
import cbor

def get_link(s: str, word: str, id: int):
    link_pos = s.find(word)
    return (id, link_pos, link_pos + len(word) - 1)

with open("text.dat", "wb") as f:
    s = "This is a little lesson in trickery, this is going down in history"
    links = [get_link(s, "lesson", 1), get_link(s, "trickery", 2), get_link(s, "history", 3)]

    offsets = [0]

    cbor.dump({
        "id": 50,
        "text": s,
        "link_mentions": links
    }, f)

    offsets.append(f.tell())

    s = "This is a little lesson in       trickery, this is going down in history"
    links = [get_link(s, "lesson", 1), get_link(s, "trickery", 2), get_link(s, "history", 3)]

    cbor.dump({
        "id": 100,
        "text": s,
        "link_mentions": links
    }, f)

tokenizer_cereal.tokenize_from_cbor_list("text.dat", "text.out2", offsets)
print(tokenizer_cereal.get_token_slice("text.out2", 0, 0, 128))