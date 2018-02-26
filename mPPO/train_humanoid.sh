echo "np 8" >> Humanoid.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 1) 2>> Humanoid.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 2) 2>> Humanoid.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 3) 2>> Humanoid.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 4) 2>> Humanoid.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 5) 2>> Humanoid.txt

echo "np 4" >> Humanoid.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 1) 2>> Humanoid.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 2) 2>> Humanoid.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 3) 2>> Humanoid.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 4) 2>> Humanoid.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 5) 2>> Humanoid.txt


echo "np 2" >> Humanoid.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 1) 2>> Humanoid.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 2) 2>> Humanoid.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 3) 2>> Humanoid.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 4) 2>> Humanoid.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 5) 2>> Humanoid.txt


echo "np 1" >> Humanoid.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 1) 2>> Humanoid.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 2) 2>> Humanoid.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 3) 2>> Humanoid.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 4) 2>> Humanoid.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Humanoid-v1 --num-timesteps 1000000 --times 5) 2>> Humanoid.txt

