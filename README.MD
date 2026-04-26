====================================================================
HE THONG NHAN DIEN BIEN SO XE TU DONG (ALPR) - UET AI SYSTEM
====================================================================

He thong su dung kien truc hoc sau (Deep Learning) hai giai doan (2-Stage Pipeline) dua tren mo hinh YOLOv10 de nhan dien bien so xe va doc ky tu thoi gian thuc.

1. KIEN TRUC HE THONG
--------------------------------------------------------------------
- Giai doan 1 (LP Detection): Su dung mo hinh YOLOv10n de phat hien va cat (crop) vi tri bien so xe trong khung hinh tu Camera/Video.
- Giai doan 2 (Character Recognition - OCR): Vung bien so sau khi cat se duoc dua vao mo hinh YOLOv10 thu hai de nhan dien tung ky tu (chu cai va con so).
- Hau xu ly (Post-processing): Thuat toan sap xep toa do (Sorting Algorithm) xy ly dac thu bien so Viet Nam (bien vuong 2 dong va bien dai 1 dong), dam bao ghep chuoi ky tu chuan xac.


2. CAU TRUC THU MUC (PROJECT STRUCTURE)
--------------------------------------------------------------------
BTL-THPTTTNT-36/
|-- .idea/                  # Thu muc cau hinh cua IDE (PyCharm)
|-- datasets/               # Du lieu huan luyen (Anh va Nhan cho LP_detection va OCR)
|-- result_train(pt)/       # Chua cac file trong so mo hinh sau khi train (lp_model.pt, ocr_model.pt)
|-- runs/                   # Chua ket qua log va bieu do sau cac dot huan luyen cua Ultralytics
|-- training/               # Thu muc chua cac script dung de huan luyen mo hinh
|-- data.yaml               # File cau hinh duong dan den dataset
|-- main.py                 # File chay he thong 2-Stage ALPR thoi gian thuc (Final)
|-- testYoloV10.py          # Script test thu nghiem mo hinh co ban
|-- requirements.txt        # Danh sach thu vien can cai dat
|-- yolov10n.pt             # Trong so goc cua YOLOv10 Nano (Pre-trained)
|-- README.txt              # Tai lieu huong dan du an


3. HUONG DAN CAI DAT THU VIEN (INSTALLATION)
--------------------------------------------------------------------
Yeu cau he thong: May tinh da cai dat san Python >= 3.9.

* Buoc 1: Mo Terminal (hoac Command Prompt / PowerShell) tai thu muc goc cua project (BTL-THPTTTNT-36).

* Buoc 2 (Khuyen nghi): Tao va kich hoat moi truong ao (Virtual Environment) de tranh xung dot thu vien voi cac project khac tren may.
  - Lenh tao moi truong ao:
    python -m venv venv
  - Lenh kich hoat moi truong ao (Tren Windows):
    venv\Scripts\activate

* Buoc 3: Cai dat cac thu vien.
  Sau khi kich hoat moi truong (co chu (venv) o dau dong lenh), chay lenh sau de he thong tu dong tai va cai dat toan bo thu vien tu file requirements.txt:
    pip install -r requirements.txt
  (Doi 1-2 phut de he thong tai Ultralytics, OpenCV, PyTorch... ve may)


4. CACH SU DUNG (USAGE)
--------------------------------------------------------------------
- De khoi dong he thong nhan dien qua Webcam, dam bao ban da dat dung file trong so (.pt) vao thu muc va chay lenh:
  python main.py

- De test thu nghiem mot chuc nang rieng le:
  python testYoloV10.py

