# Cloud Concepts

Q1 : What is the core economic model of cloud computing, and how does it differ from owning your own servers?
A : Core economic model = pay-as-you-go / rent vs. own (operating expense vs. capital expense).
cloud converts large upfront hardware purchases (capex) into on-demand rental (opex); you pay only for what you use and can scale up/down without buying servers.

Q2: What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.
A: Diffrence between vertical and horizontal scaling is:
Vertical scaling = making one machine bigger (more RAM, CPU, GPU)
Horizontal scaling = adding more machines and splitting the work across them.

Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

1. A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch - horizontal (spread load across many servers).
2. A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM - vertical (one more powerful machine).
3. A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines - horizontal (parallelize the work).

Q3: Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.
Gmail
Azure Virtual Machines
AWS S3 (Simple Storage Service)
GitHub Codespaces
Snowflake
Supabase
Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

A: Most likely intended classifications:
Gmail - SaaS (finished application)
Azure Virtual Machines - IaaS (raw compute you manage)
AWS S3 (Simple Storage Service) - IaaS (storage infrastructure)
GitHub Codespaces - PaaS (managed dev environment/platform)
Snowflake - SaaS (managed data platform; accept PaaS-leaning reasoning)
Supabase -  BaaS (backend-as-a-service) 

Iaas : You manage operating system, install your software, configure your environment, and handle security updates. This is called Infrastructure as a Service (IaaS). Example : Azure Virtual Machines.

PaaS : You manage just your app/code, provider manages the platform is called Platform as a Service (PasS). Example : Google App Engine

SaaS : You use others developed applications without worrying about any maintainance. That's Software as a Service (SaaS). Example: Gmail

Q4 : What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?
A: Managed data platform (Databricks/Snowflake) is a fully-managed analytics/data system built on top of raw cloud.
Gain: less setup/ops, built-in scaling, integrated tooling, faster time-to-value.
Give up: some control, flexibility, and cost (markup over raw infra), plus potential vendor lock-in.

Q5: The lesson describes a situation where the cloud is probably not the right choice. What is that situation?
A: If your dataset fits comfortably on a single machine and you do not have massive compute demands, local processing is often faster and cheaper. This is often the best approach when setting up an initial prototype.

# Cloud Landscape
Q1 : Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.
A : The three hyperscalers: AWS, Microsoft Azure, Google Cloud (GCP). 
AWS — broadest/most mature service catalog; startups to large enterprises.
Azure — deep Microsoft/enterprise integration; orgs already on Microsoft stack.
GCP — strength in data, analytics, and machine learning (the lesson names BigQuery and Vertex AI); data-heavy and ML-focused 

Q2 : The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?
A : Three reasons the course switched Azure→Supabase (summarize each). The lesson gives exactly these three:

Access — Azure needs organizational provisioning (joining a tenant, waiting for an invitation), which can block students for days. Students create a Supabase account themselves in minutes, and the free tier covers the whole course.
Pedagogical fit — Azure Blob Storage stores opaque files by path. Supabase stores rows and columns in a relational database, and querying structured data is a more transferable skill.
Pipeline coherence — the raw zone and enriched zone of the Weeks 9–11 pipeline become two related tables that are easy to query, inspect, and debug.
Accept the student's own wording as long as each reason maps to one of these three. Do not require the lesson's labels. Answers such as "easier signup" or "free tier" are part of access; they do not count as separate reasons. Reflection should land on: match the tool to the project's scale and the team's skill — don't default to the biggest/most complex provider.

Q3 : For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.
A : 
You need to store 10 TB of image files and retrieve them by filename from any machine - object storage → AWS S3 (or GCS / Azure Blob).
You need to run an ML training job on a GPU for four hours, then shut it down - compute / IaaS VM (GPU) → AWS EC2 (e.g. p3), GCP Compute Engine.
You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets - AWS Lambda, Google Cloud Run.
You need to send structured data to a large language model and get a text response back - LLM API → OpenAI API (or Anthropic).

Q4: The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?
A: Designed project with a multi-provider stack (e.g. S3 for storage + OpenAI for LLM + Supabase for DB). Reflection: consolidating to one provider gains simpler billing/integration/security, but gives up best-of-breed tools and increases lock-in. Any coherent stack drawing from ≥2 providers is fine.

