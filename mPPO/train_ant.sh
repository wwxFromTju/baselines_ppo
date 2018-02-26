echo "np 8" >> Ant.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 1) 2>> Ant.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 2) 2>> Ant.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 3) 2>> Ant.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 4) 2>> Ant.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 5) 2>> Ant.txt

echo "np 4" >> Ant.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 1) 2>> Ant.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 2) 2>> Ant.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 3) 2>> Ant.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 4) 2>> Ant.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 5) 2>> Ant.txt


echo "np 2" >> Ant.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 1) 2>> Ant.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 2) 2>> Ant.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 3) 2>> Ant.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 4) 2>> Ant.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 5) 2>> Ant.txt


echo "np 1" >> Ant.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 1) 2>> Ant.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 2) 2>> Ant.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 3) 2>> Ant.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 4) 2>> Ant.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Ant-v1 --num-timesteps 1000000 --times 5) 2>> Ant.txt

