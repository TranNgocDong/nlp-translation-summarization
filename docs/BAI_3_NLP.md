# Bai 3. Mo hinh Dich thuat va Tom tat Truyen/Van ban lon (NLP)

## 1. Phat bieu bai toan

Xay dung mot he thong NLP nhan vao mot van ban dai (truyen, chuong truyen, bai bao, doan mo ta su kien) va sinh ra:

- Mot ban tom tat ngan gon, giu lai y nghia chinh.
- Hoac mot `graph` the hien cac moi quan he giua nhan vat.

## 2. Bieu dien duoi dang ham `f(x) = y`

Dat:

- `x = (e_1, e_2, ..., e_n)` la day vector embedding cua van ban goc.
- `f_theta` la mo hinh hoc sau duoc huan luyen.
- `y = (y_1, y_2, ..., y_m)` la chuoi tom tat.
- Hoac `y = G = (V, E)` la do thi quan he, trong do `V` la tap nhan vat va `E` la tap canh quan he.

Khi do:

`f_theta(x) -> y`

Vi du:

`f("Doan van 10,000 chu ve tran chien") = "Nhan vat A danh bai nhan vat B va lay duoc bau vat"`

## 3. Mo hinh duoc ap dung

He thong su dung mo hinh `Transformer Seq2Seq`, cu the la `ViT5`, thuoc nhom encoder-decoder.

### Vi sao chon Transformer

- Xu ly ngu canh tot hon RNN/LSTM o van ban dai.
- Phu hop bai toan sinh chuoi dau ra moi tu mot chuoi dau vao.
- Co the fine-tune de tom tat tieng Viet va tieng Anh.

## 4. Cach xu ly van ban lon

Do `Transformer` co gioi han do dai input, he thong ap dung chien luoc tom tat nhieu tang:

1. Chia van ban dai thanh nhieu doan nho.
2. Tom tat tung doan bang `ViT5`.
3. Gop cac tom tat trung gian.
4. Tom tat lai mot lan nua de sinh ket qua cuoi.

Cach nay giup bai toan "van ban lon" khong bi cat mat noi dung o phan dau.

## 5. Dau vao va dau ra cua he thong

### Dau vao

- Mot chuoi van ban goc.
- Ngon ngu nguon: `vi` hoac `en`.
- Ngon ngu dich dich: `vi` hoac `en`.

### Dau ra

- `summary_vi`: ban tom tat tieng Viet.
- `summary_en`: ban tom tat tieng Anh.
- `translated_text`: ban dich neu co model dich trong local cache.
- `relation_graph`: do thi quan he nhan vat.

## 6. Graph quan he nhan vat

Sau khi co ban tom tat, he thong trich xuat:

- `nodes`: ten nhan vat.
- `edges`: moi quan he giua nhan vat, vi du `danh_bai`, `bao_ve`, `gap_go`, `phan_boi`.

Vi du:

- Node: `Nhan vat A`, `Nhan vat B`
- Edge: `Nhan vat A --danh_bai--> Nhan vat B`

## 7. Thanh phan trong project

- API: [server.py](/E:/dong/server.py)
- Pipeline tom tat: [pipeline.py](/E:/dong/summarization/pipeline.py)
- Wrapper model: [vit5_wrapper.py](/E:/dong/summarization/vit5_wrapper.py)
- Graph extractor: [extractor.py](/E:/dong/relation_graph/extractor.py)
- Giao dien demo: [app.py](/E:/dong/UI/app.py)

## 8. Ket luan

De tai da dap ung dung cac yeu cau chinh:

- Co ap dung mang neural sau cho NLP (`Transformer Seq2Seq`).
- Mo ta ro ham `f(x) = y`.
- Dau vao la chuoi vector bieu dien van ban.
- Dau ra la ban tom tat moi va graph quan he nhan vat.
- Ho tro xu ly van ban dai bang chien luoc chia doan va tom tat nhieu tang.
