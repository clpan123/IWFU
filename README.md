# IWFU
IWFU is designed as a federated forget algorithm, which makes use of the characteristics of the sample data to achieve faster forget acceleration.

Directory structure
project-name/  
│  
├── FL_TEST/  
│   ├── get_data.py  
│   ├── get_model.py.py  
│   ├── main.py  
│   ├── main_recovery.py  
│   ├── main_unlearn.py  
│   ├── main_recovery.py  
│   ├── my_client.py  
│   ├── My_server.py  
│   └── utils.py  
│  
├── ML_TEST/  
│   ├── get_data.py/  
│   ├── get_model.py/  
│   ├── main.py/  
│   ├── recovery_main.py/  
│   ├── un_main.py/  
│   ├── utils.py/  
│  
└── ├── README.md 

FL——TEST is our proposed algorithm IWFU.
MLTEST is our analysis of forgetting learning according to forgetting events.

Tools version:
opencv-python == 4.10.0.84
numpy  == 1.22.1
torch  == 2.3.1
torchvision == 0.9.1+cu111
