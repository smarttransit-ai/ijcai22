# Offline Vehicle Routing Problem with Online Bookings: A Novel Problem Formulation with Applications to Paratransit

This repository describes the results of the code and the data for the following paper: https://smarttransit.ai/files/sivagnanam2022offline.pdf
### ABSTRACT

*Vehicle routing problems (VRPs) can be divided  into two major categories: offline VRPs, which
consider a given set of trip requests to be served,
and online VRPs, which consider requests as they
arrive in real-time. Based on discussions with public transit agencies, we identify a real-world problem that is not addressed by existing formulations:
booking trips with flexible pickup windows (e.g., 3
hours) in advance (e.g., the day before) and confirming tight pickup windows (e.g., 30 minutes) at
the time of booking. Such a service model is often required in paratransit service settings, where
passengers typically book trips for the next day
over the phone. To address this gap between offline and online problems, we introduce a novel formulation, the offline vehicle routing problem with
online bookings. This problem is very challenging computationally since it faces the complexity
of considering large sets of requests—similar to offline VRPs—but must abide by strict constraints on
running time—similar to online VRPs. To solve
this problem, we propose a novel computational
approach, which combines an anytime algorithm
with a learning-based policy for real-time decisions. Based on a paratransit dataset obtained from
the public transit agency of Chattanooga, TN, we
demonstrate that our novel formulation and computational approach lead to significantly better outcomes in this setting than existing algorithms.*

### ACKNOWLEDGEMENT

This material is based upon work sponsored by the National
Science Foundation under Grant CNS-1952011 and by the
Department of Energy under Award DE-EE0009212.

**Disclaimer**: Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not
necessarily reflect the views of the National Science Foundation or the Department of Energy.

### Directories

[algo](algo):
contains implementation of,
#### Our Algorithms
1. [Online Booking Simulator](algo/heuristic/OnlineBookingSimulator.py)
2. [Deterministic Greedy](algo/heuristic/GreedyPTOpt.py)
3. [Simulated Annealing](algo/heuristic/SABasedPTOpt.py)

####  Wrapper for other existing baselines
1. [Google or-tools](https://developers.google.com/optimization)
2. [VRoom](http://vroom-project.org/) .

[base](base): contains the custom implementation to solve the para-transit optimization problem.

[common](common): constants and common utilities shared by source files in other folders ([algo](algo), [base](base)).

[data](data): contains the all data to run paratransit sample instances.


#### Data Folder Description:
**Main Input Data**
[data/AGENCY_A/base/para_transit_trips_2021.csv](data/AGENCY_A/base/para_transit_trips_2021.csv ) - 180 days of requests data

#### Trained Models
1. [data/AGENCY_A/models/agent_with_anytime_model.h5](data/AGENCY_A/models/agent_with_anytime_model.h5) - trained (with anytime) RL agent model.
2. [data/AGENCY_A/models/agent_with_anytime_weights.h5](data/AGENCY_A/models/agent_with_anytime_weights.h5) - trained (with anytime) RL agent model weights.
3. [data/AGENCY_A/models/agent_without_anytime_model.h5](data/AGENCY_A/models/agent_without_anytime_model.h5) - trained (without anytime) RL agent model.
4. [data/AGENCY_A/models/agent_without_anytime_model.h5](data/AGENCY_A/models/agent_without_anytime_model.h5) - trained (without anytime) RL agent model weights.

#### TRAINING AND EVALUATION

I) Normal Environment Setup (supports Linux and MacOSX):

Please make sure the following before executing the scripts
1. python 3.8
2. Install all modules listed in "requirements.txt" (using the command pip install -r requirements.txt).
3. Execute ```add_zip_codes.py``` to add retrieve zip codes for pickup and dropoff locations 
   (this can take significant time (900 minutes for full data, with 25k trip requests),
   if you try with full data; so use small set of data, i.e., filter by date)
4. Execute ```generate_travel_matrices.py``` to add travel time matrices and travel distance matrices
   (this can take significant time, if you try with full data; so use small set of data, i.e., filter by date)

Sample Training:

1. Train policy without anytime algorithm in between, and Run VRoom
```bash
python3 train_agent.py --train_env_type=agent_without_anytime --random_seed=0
```

2. Train policy with anytime algorithm in between, and Run VRoom
```bash
python3 train_agent.py --train_env_type=agent_with_anytime --random_seed=0
```

Allowed Parameters (Training):
```train_env_type``` - indicates the environment type
possible values:
    ```agent_without_anytime``` - training without anytime algorithm in between
    ```agent_with_anytime``` - training with anytime algorithm in between

```anytime_algo``` - indicates anytime algorithm that used to train the policy.

```random_seed``` - random_seed to select sample days

```no_of_episodes``` - number of episodes (number of days

```anytime_plan_duration``` - time to run anytime algorithm in seconds during deciding tight pickup windows

```plan_duration``` - time to run anytime algorithm in seconds at the end of the day in online simulator or 
running simulated annealing alone


Sample Execution:

sample chains with date ```0``` - ```179```

1. Greedy Algorithm

```bash
python3 run_optimizer.py --algo=greedy --date=0
```

2. Simulated Annealing Algorithm

```bash
python3 run_optimizer.py --algo=sim_anneal --date=0
```

3. Online Booking with Agent Trained without anytime + Simulated Annealing at the End

```bash
python3 run_optimizer.py --algo=agent_without_anytime --date=0
```
 
4. Online Booking with Agent Trained with anytime in between + Simulated Annealing at the End

```bash
python3 run_optimizer.py --algo=agent_with_anytime --date=0
```

5. VROOM

```bash
python3 run_optimizer.py --algo=vroom --date=0
```

6. Google OR-Tools Routing

```bash
python3 run_optimizer.py --algo=routing --date=0
```

Allowed Parameters (Execution):

```algo``` - indicates the name of algorithm

```anytime_algo``` - indicates anytime algorithm that used to train the policy.

```date``` - indicates the date

```anytime_plan_duration``` - time to run anytime algorithm in seconds during deciding tight pickup windows

```plan_duration``` - time to run anytime algorithm in seconds at the end of the day in online simulator or 
running simulated annealing alone
