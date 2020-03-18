Cài đặt :
    Python 3.5,
    Numpy 1.14, 
    TensorFlow 1.8, 
    win32com, 
    scipy, 
    gensim 
    matplotlib, 
    json, 
    pickle

Sử dụng:
    - Run demo.py: Sinh mô tả cho các ảnh trong folder images
	- Run test_val.py: Sinh mô tả cho test case trong dataset
    - Muốn sinh mô tả cho câu riêng paste ảnh vào folder images, run caption_image.py
	- File extract_feautures.py mã hóa ảnh
	- File material.py mã hóa câu
	- File modal chứa modal cho mô hình
	- File reference sinh các câu reference cho tính độ đo BLEU, METEOR
	- File train.py thực hiện huấn luyện
Chỉnh sửa:
	- Thay dataset sửa trong file material.py
	- Thay số vòng chạy, beam size sửa trong file ultils.py
folder pretrained chứa word2vec
folder result_test chứa kết quả của test dùng để đo BLEU
folder weight_model chứa weight_model cho mô hình.
# Image-Captioning-With-Deep-Learning-and-Beam-Search
