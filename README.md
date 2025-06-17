<!-- <div align="center">
    <img src="./assets/macrocosmos-black.png" alt="Alt generative-folding-tao">
</div> -->

<picture>
    <source srcset="./assets/macrocosmos-white.png"  media="(prefers-color-scheme: dark)">
    <img src="macrocosmos-white.png">
</picture>

<picture>
    <source srcset="./assets/macrocosmos-black.png"  media="(prefers-color-scheme: light)">
    <img src="macrocosmos-black.png">
</picture>

<div align="center">

</div>


<div align="center">
    <img src="./assets/mainframe_official.png" alt="mainframe-official">
</div>

*inspiration from [owl_posting](https://x.com/owl_posting)*

<div align="center">

# Mainframe

</div>

Subnet 25, Mainframe (formerly known as Protein-Folding) is the first decentralized science subnet on Bittensor. Its focus is on creating DeSci technology for pharmaceutical companies, researchers, and academics. Presently, its focus is on using global computing power to simulate protein molecular dynamics via OpenMM, and protein-ligand docking using DiffDock; both processes essential in most drug discovery pipelines. However, this subnet is designed to be adaptive to a wide array of computing-based problems in the life sciences, utilizing Bittensor’s tokenomics and incentive structure to offer affordable solutions.

Mainframe asks a simple question: can a decentralized, incentivized pool of people be used for generalized scientific compute? At Macrocosmos, we believe it is possible. Not only that, we believe that decentralized and accessible in-silico experimentation is imperative to accelerate science.

"Rational simulation-guided design of atomic systems has been a dream of researchers across the chemical sciences for decades. Enabling rapid and performant experimentation to experts would unlock massive potential to accelerate chemical science" [Mann et al. 2025](https://rowansci.com/publications/egret-1-pretrained-neural-network-potentials)

Mainframe attempts to tackle this very important challenge. 



<div align="center">

<a href="https://app.macrocosmos.ai/mainframe">
  <img src="./assets/mainframe-link.png" alt="mainframe" width="300"/>
</a>

👆🏼 enter the mainframe app 👆🏼

</div>

<div align="center">

## Real-World Impact

</div>

✅ Subnet 25 has partnered with Rowan Scientific, an AI-based chemistry software developer. Together, we’re building next generation neural network potential (NNPs) models via specialized DFT data generation on Mainframe. By collaborating, we’re helping create more powerful drug discovery tools for researchers and academics. 

### 🤝 Read more about our collaboration here:
1. [Partnership Announcement](https://www.rowansci.com/blog/partnering-with-macrocosmos?utm_source=substack&utm_medium=email)
2. [Macrocosmos & Rowan in Forbes](https://www.forbes.com/sites/torconstantino/2025/05/14/this-decentralized-ai-could-revolutionize-drug-development/)

### 🧑🏻‍💻 Mainframe App: 
https://app.macrocosmos.ai/mainframe

### 📖 Mainframe Documentation: 
https://docs.macrocosmos.ai/subnets/subnet-25-mainframe 

### 👾 Mainframe’s API: 
https://docs.macrocosmos.ai/developers/api-documentation/sn25-mainframe 

### 🧮 Mainframe's Protein Folding Dashboard: 
https://www.macrocosmos.ai/sn25/dashboard


<div align="center">

# Subnet statistics

</div>

This subnet has the following statistics backing it, which show its utility and efficiently:

- 162,200 proteins folded since launch 📈
- Simulation speed of approximately 132,000 nsec/day 🏎️
- Approximately 17 petaflops (20% more than Folding@Home)

<div align="center">

# Core Team

</div>

**Will Squires**, Co-Founder and CEO: Will is an entrepreneur with experience building AI startups. Will has a Master of Engineering degree in civil engineering and sustainability from the University of Warwick.

**Steffen Cruz**, Co-Founder and CTO: Steffen formerly worked at OpenTensor Foundation (the organization behind Bittensor) as their CTO, and is the original architect of Subnet 1, Apex’s codebase. Steffen has a PhD in Experimental Nuclear Physics from the University of British Columbia, Canada.

**Brian McCrindle**, Subnet Lead and Founding Engineer: Brian is the leading subnet designer for Mainframe. Brian formerly worked at OpenTensor Foundation as a Machine Learning Researcher. He has an MASc. in Electrical and Computer Engineering and Computer Vision from McMaster University.

**Syzmon Fonau**, Senior Machine Learning Engineer: Szymon works on expanding and refining subnet 25, making it more robust and functional. He has three years of experience in leadership positions at the Director/C-suite level.

<div align="center">

# Market

</div>

Subnet 25’s product-market-fit is within the following customer groups:

1. Academia: Biochemistry, structural biology, computational chemistry, bioinformatics, or pharmaceutical sciences

2. Governments and non-profit organizations: Institutions or national labs focused on health, biotech, or bioinformatics

3. Small labs and startups: Small to mid-sized biotech startups looking to outsource or speed up early-stage discovery

4. Contract research organizations: Service providers doing research for pharma clients.

5. Pharma and biotech companies: In-house drug discovery teams working on lead identification and optimization

This is because Mainframe offers low-cost solutions that many smaller organizations typically would struggle with access to. This is especially the case when it comes to protein-docking via molecular dynamics, which can be costly due to computing power.

<div align="center">

# Mainframe Incentive Architecture

</div>

As Mainframe is meant to be a collection of different scientific computation tasks happening in parallel, the result is that there are a collection of incentive mechanisms that must be managed. The sections below outline our current tasks: 

<div align="center">

### Molecular Dynamics 🧬
---

</div>

Physical systems such as proteins tend to minimize their energy and so this provides a succinct, exploit-resistant and highly sensitive measure of quality. For this reason, miners compete to provide protein configurations that coincide with the lowest energy (analogous to loss). The benefit is twofold; the metric the network is optimizing for is highly aligned with the desired outcome (biologically stable structures) and it is transparent and deterministic (both miners and validators can quickly calculate the energy of a configuration).

Miners are currently tasked with finding the lowest free energy solution of the specified configuration. This challenge is placed within the Global Job Pool (GJP) and miners must work on the challenge and post their solutions to S3 for validation. Therefore, all miners have access to all challenges, and thus compete on them accordingly. 

The top-K miners (currently, K = 5) are ranked, where 80% of the reward for that challenge is given to the top miner in the batch, and the remaining 20% is distributed to the (K-1) miners. Therefore, miners not in the top-K set get *no* rewards. 

**Maintaining opportunity:**
-  The miners are oversubscribed to jobs by design, which means there is an effectively unbounded opportunity for those that can handle the enormous computational workload.

**Ensuring innovation:** 
- Each miner uses a separate random seed for their simulations, which ensures that each simulation suitably explores the folding space and utilizes the parallelism opportunity of batching jobs. On job evaluation, if miners are submitting identical results, we enforce that their reward is zero, which continues to incentivize unique solutions.

<div align="center">

### Density Functional Theory (DFT) ⚛️
---

</div>

Coming soon...

<div align="center">

# Subnet Future Value

</div>

Mainframe is poised to become more valuable as it’s being upgraded into a general purpose life-sciences subnet. Its expansion in Q1 2025 from being purely a protein-folding subnet to a tool that can handle a wide range of unique jobs. 

### Use-cases and potential

Mainframe, has several use-cases:
1. Aid in the drug discovery and design process with its protein-folding and docking toolkit.
Use molecular dynamics to generate potential protein and ligand combinations. that could be used to further academic research at a low cost.

2. Help solve computational problems in the life-sciences sector by offering organizations a global pool of computing resources.

3. Since the pivot in 2025 towards the entire life-sciences domain, Mainframe is able to plug into a range of problems in the scientific world. Its aim is to become one of many tools used by researchers to forward their work, and further help save lives.

4. Mainframe’s first collaboration and partnership with Rowan Scientific is an example of how we can support STEM. By helping provide density-functional-theory (DFT) configurations to their work, we’re actively involved in building greater chemical models.
Dashboards, tools, resources

<div align="center">

# Dashboards, Tools, and Additional Resources

</div>

### [Molecular Dynamics Dashboard](https://www.macrocosmos.ai/sn25/dashboard)
Interactive charts surface validator–miner weights, throughput and energy‑efficiency metrics so you can verify that incentives are aligned with high‑quality outputs rather than empty computation, while cohort‑level benchmarking highlights the subnet’s cost‑per‑fold advantage over centralised alternatives. 

### [Macrocosmos SDK](https://github.com/macrocosm-os/macrocosmos-py)
Macrocosmos has a public SDK that provides you with API endpoints for using all of our products! 

# License

This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2024 Macrocosmos AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
