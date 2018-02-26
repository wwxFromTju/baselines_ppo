echo "np 8" >> Walker2d.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 1) 2>> Walker2d.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 2) 2>> Walker2d.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 3) 2>> Walker2d.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 4) 2>> Walker2d.txt
(time mpirun -np 8 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 5) 2>> Walker2d.txt

echo "np 4" >> Walker2d.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 1) 2>> Walker2d.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 2) 2>> Walker2d.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 3) 2>> Walker2d.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 4) 2>> Walker2d.txt
(time mpirun -np 4 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 5) 2>> Walker2d.txt


echo "np 2" >> Walker2d.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 1) 2>> Walker2d.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 2) 2>> Walker2d.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 3) 2>> Walker2d.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 4) 2>> Walker2d.txt
(time mpirun -np 2 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 5) 2>> Walker2d.txt


echo "np 1" >> Walker2d.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 1) 2>> Walker2d.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 2) 2>> Walker2d.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 3) 2>> Walker2d.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 4) 2>> Walker2d.txt
(time mpirun -np 1 python -m myppo.ppo.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 5) 2>> Walker2d.txt

