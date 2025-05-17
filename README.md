# 🎮 Learning to Play Super Mario Bros using DDQN with ResNet-18 and SWIN Transformer


This repository presents a deep reinforcement learning (DRL) approach to train an autonomous agent that learns to play Super Mario Bros from raw pixel data using a hybrid deep neural network architecture combining ResNet-18 and SWIN Transformer, coupled with Double Deep Q-Networks (DDQN).

The architecture is designed to tackle the complexity of temporally dependent, pixel-based decision-making under partial observability in dynamic environments. Our model demonstrates faster convergence and superior performance over traditional CNN-based agents.

---

## 🧠 Motivation
Traditional DRL methods like DQN and vanilla CNNs fall short when facing:

* Temporal dependencies between frames

* Large visual state spaces

* Instabilities in Q-value estimation

To address these, we propose a DDQN-based agent with ResNet-18 for deep spatial feature encoding and SWIN Transformer for attention-based fine-grained decision-making. Additionally, we reduce the action space to significantly enhance convergence speed and learning reliability.

---

## 🛠️ Architecture
### 🧮 Algorithm
* Double Deep Q-Network (DDQN) — mitigates Q-value overestimation via separate online and target networks.

* Epsilon-Greedy Policy — balances exploration and exploitation with decay.

* Experience Replay — decorrelates samples for stable learning.

* Target Network Updates — delayed synchronization improves stability.

### 🧠 Neural Network Stack

<pre> 
  Stacked 4 grayscale frames (4 × 64 × 64) 
                 ↓
  ResNet-18: Encodes spatial features
                 ↓ 
  SWIN Transformer: Captures inter-frame attention via shifted windows 
                 ↓
  Fully Connected Head: Outputs Q-values for 5 discrete actions 
  </pre>


### 🧩 Action Space
Reduced from 256 native actions → 5 essential discrete actions:

* ```Right```

* ``` Jump```

* ```Jump + Right```

* ``` Sprint + Right```

* ```No-op```

This minimalistic action space was critical to stabilizing early-stage exploration and enabling fast convergence.

---

## 🧪 Environment & Preprocessing
* Platform: ``` gym-super-mario-bros v0 ```

* Wrappers Used:

  * ```FrameSkip```: Skips 4 frames per action

  * ```Grayscale & Resize```: Converts 240x256 RGB → 64x64 grayscale

  * ```FrameStack```: Maintains a buffer of last 4 frames for temporal awareness

* Observation Shape: ```(4, 64, 64)```
---
## 🔍 Training Procedure
### Memory Management
* Experience Replay Buffer: deque(maxlen=100_000)

* Batch Size: 64

* Learning Rate: 1e-4

* Discount Factor (γ): 0.99

* Update Target Network: every 10 episodes

### Exploration Strategy
Epsilon Decay: from 1.0 to 0.05 linearly over 50,000 steps

---

## 📈 Results
| __Model__ | __Avg Reward @ 1.25k epiosdes__|__Avg Reward @ 10k episodes__|
|:---:|:---:|:---:|
|DDQN + SWIN + ResNet| 2700+| >3000|
|CNN + DDQN(baseline)|~800|~2250|

* Our model achieved over 2,700 average reward in just 1,250 episodes, outperforming the baseline CNN+DDQN that needed 10k episodes to reach 2,250.
* The SWIN Transformer’s shifted windows enhanced representation learning by enabling localized self-attention while preserving computational efficiency.
* The below GIF shows the agent playing the game after 1,250 episodes

<p align="center">
  <img src="Simulation_1250episodes.gif" width="70%" alt="Demo GIF" />
</p>


---


## 🚧 Future Work
* Extend to multi-agent scenarios (e.g., adversarial Mario)

- Use prioritized experience replay for more effective memory sampling

* Hyperparameter search for SWIN-specific configurations

* Extend to 3D visual navigation and robotics control environments
