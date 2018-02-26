echo "hopper" >> hopper.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Hopper-v1 --num-timesteps 1000 --times 1) 2>> hopper.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Hopper-v1 --num-timesteps 1000 --times 2) 2>> hopper.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Hopper-v1 --num-timesteps 1000 --times 3) 2>> hopper.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Hopper-v1 --num-timesteps 1000 --times 4) 2>> hopper.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Hopper-v1 --num-timesteps 1000 --times 5) 2>> hopper.txt


echo "Walker2d" >> Walker2d.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000 --times 1) 2>> Walker2d.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000 --times 2) 2>> Walker2d.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000 --times 3) 2>> Walker2d.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000 --times 4) 2>> Walker2d.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000 --times 5) 2>> Walker2d.txt

echo "Ant" >> Ant.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000 --times 1) 2>> Ant.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000 --times 2) 2>> Ant.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000 --times 3) 2>> Ant.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000 --times 4) 2>> Ant.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000 --times 5) 2>> Ant.txt

echo "Swimmer" >> Swimmer.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000 --times 1) 2>> Swimmer.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000 --times 2) 2>> Swimmer.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000 --times 3) 2>> Swimmer.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000 --times 4) 2>> Swimmer.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000 --times 5) 2>> Swimmer.txt

echo "Humanoid" >> Humanoid.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000 --times 1) 2>> Humanoid.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000 --times 2) 2>> Humanoid.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000 --times 3) 2>> Humanoid.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000 --times 4) 2>> Humanoid.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000 --times 5) 2>> Humanoid.txt

echo "HumanoidStandup" >> HumanoidStandup.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env HumanoidStandup-v1 --num-timesteps 1000 --times 1) 2>> HumanoidStandup.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env HumanoidStandup-v1 --num-timesteps 1000 --times 2) 2>> HumanoidStandup.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env HumanoidStandup-v1 --num-timesteps 1000 --times 3) 2>> HumanoidStandup.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env HumanoidStandup-v1 --num-timesteps 1000 --times 4) 2>> HumanoidStandup.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env HumanoidStandup-v1 --num-timesteps 1000 --times 5) 2>> HumanoidStandup.txt