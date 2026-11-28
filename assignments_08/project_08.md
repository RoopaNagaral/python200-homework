# project is set up 

# Part B

# Scenario A — Lightweight compute: A t3.micro EC2 instance (1 vCPU, 1 GB RAM), on-demand pricing, running 8 hours per day, 5 days per week (approximately 160 hours per month). Use the US East (N. Virginia) region.

# Cost Estimation A - 
Scenario A: ~$1.66/month


# Scenario B — Heavy analytics workload: A p3.2xlarge EC2 instance (8 vCPU, 1 V100 GPU), running 24/7 for the full month (730 hours); an RDS db.m5.large instance (2 vCPU, 8 GB RAM); and an S3 Standard storage bucket with 1 TB of data. Use US East (N. Virginia).

# Cost Estimation B - 
Scenario B: ~$2,383/month

Difference: Scenario B is over 1,400 times more expensive than Scenario A.

# A sentence on how the two scenarios compare — what does the cost difference tell you about when a GPU instance is or isn't worth it?

The large cost difference shows that GPU instances are only worth it for specialized workloads such as machine learning, AI training, or large-scale analytics, where the extra performance justifies the much higher expense. For everyday applications, testing, small websites, or development work, a low-cost EC2 instance is usually the more cost-effective choice.

# Video link
# https://youtu.be/0S4VFx2Snpo