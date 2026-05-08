def load_and_chunk():
    with open(r"C:\Users\VICTUS\Downloads\LLLM-RAG-\LLM\data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    chunks = []
    current_section = ""

    for line in lines:

        # 🔹 key:value หลัก
        if ":" in line and not line.startswith("-"):
            chunks.append(line)

        # 🔹 header (ไม่มี :)
        elif not line.startswith("-"):
            current_section = line

        # 🔹 bullet
        elif line.startswith("-"):
            content = line[1:].strip()

            if ":" in content:
                sub_key, sub_val = content.split(":", 1)
                chunks.append(f"{current_section} {sub_key.strip()} : {sub_val.strip()}")
            else:
                chunks.append(f"{current_section} : {content}")

    return chunks