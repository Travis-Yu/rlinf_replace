```
rlinf_franka/                     # 主路径
├── gello_software/               # 目标：替换其下文件
│   └── gello/
│       ├── agents/               # 本仓库 gello_agent.py 替换
│       │   └── gello_agent.py
│       └── dynamixel/            # 本仓库 driver.py 替换
│           └── driver.py
├── gello-teleop/                 # 目标：替换其下文件夹
│   └── gello_teleop/             # 本仓库 gello_teleop 文件夹替换
│       └── …                     # （内部内容保留，此处省略）
└── RLinf/                        # 独立仓库，不替换，完整保留
    └── …                         # （内部结构完整，此处省略）
```
