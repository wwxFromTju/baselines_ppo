echo "np 8" >> Swimmer.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 1) 2>> Swimmer.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 2) 2>> Swimmer.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 3) 2>> Swimmer.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 4) 2>> Swimmer.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 5) 2>> Swimmer.txt

echo "np 4" >> Swimmer.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 1) 2>> Swimmer.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 2) 2>> Swimmer.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 3) 2>> Swimmer.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 4) 2>> Swimmer.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 5) 2>> Swimmer.txt


echo "np 2" >> Swimmer.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 1) 2>> Swimmer.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 2) 2>> Swimmer.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 3) 2>> Swimmer.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 4) 2>> Swimmer.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 5) 2>> Swimmer.txt


echo "np 1" >> Swimmer.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 1) 2>> Swimmer.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 2) 2>> Swimmer.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 3) 2>> Swimmer.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 4) 2>> Swimmer.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 5) 2>> Swimmer.txt

