# Meta Information

- URL: [LLMのための強化学習手法 2025 -PPO・DPO・GRPO・DAPO一気に理解する-｜olachinkei](https://note.com/olachin/n/n9706c13c8678?sub_rt=share_b)
- note guideline: [コミュニティガイドライン – noteヘルプセンター](https://www.help-note.com/hc/ja/articles/4409925863193-%E3%82%B3%E3%83%9F%E3%83%A5%E3%83%8B%E3%83%86%E3%82%A3%E3%82%AC%E3%82%A4%E3%83%89%E3%83%A9%E3%82%A4%E3%83%B3)
    - [創作を後押しする著作権の考え方 – noteヘルプセンター](https://www.help-note.com/hc/ja/articles/4409701626393-%E5%89%B5%E4%BD%9C%E3%82%92%E5%BE%8C%E6%8A%BC%E3%81%97%E3%81%99%E3%82%8B%E8%91%97%E4%BD%9C%E6%A8%A9%E3%81%AE%E8%80%83%E3%81%88%E6%96%B9)

# Overview

This markdown describes summary about the article to remind me of its content.

---

# Introduction

LLM uses reinforcement learning (RL) to improve performance after pre-training.

RLHF (Reinforcement Learning with Human Feedback) is a common method from [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html).

There is some RL methods for LLMs: TRPO, PPO, DPO, GRPO, and DAPO.

- TRPO: [[1502.05477] Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- PPO: [[1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- DPO: [[2305.18290] Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- GRPO: [[2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- DAPO: [[2503.14476] DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)

based on DQN: [[1312.5602] Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

# Transition from Value-based to Policy-based Methods

DQN is a value-based method that learns the action-value function $q_{\pi}(s_t,a_t)$.
However, it is weak for time-cotinuous action spaces.

Policy-based methods directly learn the policy $\pi(a_t/s_t)$.
It can handle time-continuous action spaces and learn stochastic policies.

# Policy Gradient Methods

- REINFORCE: [Simple statistical gradient-following algorithms for connectionist reinforcement learning | Machine Learning](https://link.springer.com/article/10.1007/BF00992696)
    - Demerit: It cannot converge stably and efficiently.
- Actor-Critic Methods: [Neuronlike adaptive elements that can solve difficult learning control problems | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/6313077)
    - Demerit: It is hard to tune the balance between bias and variance ( Policy Collapse ).
- TRPO: [[1502.05477] Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
    - Demerit: It is complex and computationally expensive.
- PPO: [[1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
    - No Demerit mentioned.
- GRPO: [[2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - Merit: It can optimize LLMs effectively with less computational resources.
- DAPO: [[2503.14476] DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
    - Merit: More efficient and stable than GRPO.

The following methods is far different from the above.

- DPO: [[2305.18290] Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
    - This seems like supervised learning rather than RL.
