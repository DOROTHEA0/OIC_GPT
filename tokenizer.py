

class OICTokenizer:
    def __init__(self):
        pass

    def build_vocab(self):
        pass

    def load_vocab(self):
        pass

    def decode(self, str_indices):
        decoded_str = ""  # 初始化一个空字符串，用于存储解码后的字符
        try:
            for code in str_indices:
                # 移除每个字符编码前的0值
                valid_bytes = [byte for byte in code if byte != 0]
                # 将有效字节转换为bytes对象
                byte_sequence = bytes(valid_bytes)
                # 解码UTF-8编码的字节序列为字符串，并追加到结果字符串中
                decoded_str += byte_sequence.decode('utf-8')
        except UnicodeDecodeError:
            print(str_indices)
        return decoded_str


    def encode(self, s):
        str_indices = []
        str_i_un_shifted = []
        for c in s:
            encoded_c = c.encode('utf-8')
            # hex_representation = [f"{byte:02x}" for byte in encoded_string]
            prefix = [0] * (4 - len(encoded_c))
            str_indices.append([b + i * 256 for i, b in enumerate(prefix + [byte for byte in encoded_c])])
            str_i_un_shifted.append(prefix + [byte for byte in encoded_c])
        return str_indices, str_i_un_shifted





def decode(hex_representation):
    # After the reset, let's redo the decoding process from the hexadecimal representation to a string using UTF-8.

    # Convert the hex representation to bytes
    encoded_bytes = bytes.fromhex(hex_representation)

    # Decode the bytes back to a string using UTF-8 encoding
    decoded_string = encoded_bytes.decode('utf-8')

    return decoded_string


if __name__ == '__main__':
    # t = AutoTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf", trust_remote_code=True)
    # # with open("vocab.txt", 'w', encoding='utf-8') as f:
    # print(t("茄子"))

    # Let's encode a string into its UTF-8 encoded form and then display each byte's hexadecimal representation.
    t = OICTokenizer()
    string_to_encode = '衄'
    print(t.encode(string_to_encode)[0])


