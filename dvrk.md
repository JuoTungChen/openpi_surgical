

# Guide to run code
We use a server-client architecture to run the code. The client is responsible for sending the images and robot kinematics to the server and the server will use the current image observation to predict robot actions. The server will then send the predicted actions back to the client to execute on the robot.

## Server
First, you will need to ssh into this machine:
```bash 
ssh iulian@10.162.34.202
```
The password is 123!@#qwe

Next, make sure the model checkpoint is downloaded and extracted to the checkpoints folder:
```bash
cd ~/Downloadds
tar -xzf {checkpoint_name}.tar.gz -C ../chole_ws/src/jhu_pi0/checkpoints/{new_model_name}/
```

Then go to this directory:
```bash
 cd ~/chole_ws/src/jhu_pi0
```

Make sure the git repository is up to date:
```bash
git fetch
```

Use the spreadsheet to check which branch to run the model. 
You can check existing local branches with:
```bash
git branch
```

Use this to go to specific branch:
```bash
git checkout {branch_you_want_to_checkout}
```

If there are new branches, you can checkout the new branch:
```bash
git checkout -b {local_branch_name} {branch_you_want_to_checkout}
```
Then, you can run the server:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py policy:checkpoint --policy.config={policy_config_name} --policy.dir={checkpoint_directory}
```

### The old model with multi-view
To run the old model with multi-view, you can use the following command:
```bash
git checkout dot_2.0

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py policy:checkpoint --policy.config=final_suturing_lre_4_wd_0_1_dot --policy.dir=../openpi/checkpoints/pi0_final_suturing_lre_4_wd_0_1_dot/7500/
```

### The new model with endoscope view only
To run the new model with endoscope view only trained on all task, you can use the following command:
```bash
git checkout mono

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py policy:checkpoint --policy.config=mono_all_data --policy.dir=./checkpoints/mono_all/9999
```

## Client
On the dvrk computer, use the grapes account, and the password is imerselab.
launch the robot and run all the ros code as usual. 

Run this to start the GUI:
```bash
python ~/catkin_ws/src/openpi_surgical/scripts/ui.py
```
Run this to be able to move the robot with the GUI:
```bash
python ~/catkin_ws/src/skay_jhu_private/src/act/move_robot.py
```

Then, you can run the client code to connect to the server and start sending images and robot kinematics:


### The old model with multi-view
For the old model with multi-view, you can run the following command:
```bash
python policy_client_nvidia.py.py --no-states
```

### The new model with endoscope view only
For the endoscope view only model, you can run the following command:
```bash
 python policy_client_mono.py --no-states
```



