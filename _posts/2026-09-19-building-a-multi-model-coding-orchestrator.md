---
title: "The Summer We Built an AI Software Factory"
published: false
---

**Project:** Work Automation

---

At the end of April, I started a personal project to stop myself from spending all day babysitting coding agents. Four months later, it had become a fully automated software factory: a team of AI agents that could research a task, argue over the plan, write the code, test it, watch the build, learn from its mistakes, and remember those lessons for the next assignment.

My summer intern, Harper Austin, then put the system through a controlled study. In our internal SWE-bench Pro experiment, the best tested configuration improved performance by more than 10% over its single-agent comparison.

Then GitHub announced [Project HydraFusion](https://github.blog/ai-and-ml/github-copilot/project-hydrafusion-frontier-quality-via-multi-model-orchestration/), a research preview built around several of the same ideas.

This is the story of Work Automation.

---

## How It Started: I Was Managing the Agents

GitHub Copilot CLI had earned a permanent place in my daily workflow. Give it one well-scoped task and my full attention, and it was genuinely great.

My actual workday rarely looked like that.

I might have one work item that needed code research, another waiting for an implementation, and a third stuck in CI. I wanted to run those tasks in parallel, sometimes in the same repository. Instead, I was juggling terminal windows, branches, plans, builds, and pull requests.

The agents were supposed to save me time, but I had become their project manager.

My colleague Johnny Leek had built an initial working orchestration template that gave me a starting point. I called my project **Work Automation** and began adapting it around my daily GitHub Copilot CLI workflow.

The first commit landed on April 30: 43 files and roughly 5,700 lines of prompts, agent definitions, repository guides, configuration, and control-plane scripts. So much for starting small.

## Building the Team

The first major change was to stop asking one agent to do everything.

I split the work into roles:

- **The researcher** investigated the task and codebase.
- **The planner** turned that research into a step-by-step approach.
- **The verifier** tried to find holes in the plan.
- **The implementer** wrote the code.
- **The validator** checked the result without being allowed to quietly fix it.
- **The build watcher** followed the pull request through CI.
- **The post-mortem agent** reviewed the entire run afterward.

It was less like hiring one brilliant engineer and more like assembling a small software team.

![The Work Automation pipeline moves through research, planning, independent review, implementation, validation, a pull request, build watching, and a post-mortem](/assets/images/work-automation/pipeline.svg)

Each stage left behind a durable handoff, and the pipeline kept a ledger of what was queued, running, blocked, or finished.

That part was not flashy, but it mattered. Agent sessions end. Terminals close. Processes crash. I wanted to reopen the project the next morning and know exactly where every task had stopped.

## Giving Every Agent Its Own Office

Imagine five engineers sharing one desk, all reaching for the same keyboard and phone at once.

Two agents could have separate branches and still collide when they tried to run the same service on the same port. They could overwrite build files, share local dependencies, or leave behind processes that confused the next task.

The solution was to give every agent team its own office.

They could check out the same repository, build the same service, and even use the same port numbers without seeing one another. When one pipeline broke its environment, the others kept working.

For smaller projects, separate copies of the code were enough. Larger services needed something stronger. My colleague John Vicondoa had built a Docker-in-Docker development system that gave each pipeline a sealed-off computer of its own. John was also one of the key people developing the broader software-factory ideas that shaped this project.

At that point, I realized I was no longer building a clever chain of prompts. I was building the office building too.

## Teaching the Agents to Argue

The verifier was originally just a separate role. That was useful, but it could still share the same blind spots as the planner if both roles used the same model.

I began sending the same plan to multiple model families—Claude, GPT, Codex, and Gemini—and asking each one to review it independently.

I did not use majority voting.

If three reviewers approved a plan and one found a real problem with evidence, I cared about the one dissenter. In code review, a valid bug does not stop being valid because the other reviewers missed it.

One well-supported objection was enough to send the plan back for revision.

This paid off on a task where the plan concluded that no code change was required. The first reviewer approved it. When I reran the plan through the full set of models, a later reviewer discovered that one of its key assumptions was false.

The plan looked safe because it changed zero files. In reality, proving that nothing needs to change can be harder than reviewing a patch.

After that, any "no change required" conclusion automatically received the highest level of review.

## The Reflexion Idea

Think about how people actually learn: we try something, someone points out what went wrong, and ideally we remember not to repeat it tomorrow.

The 2023 paper [*Reflexion*](https://arxiv.org/abs/2303.11366) showed how an AI agent could use that same pattern. It attempts a task, receives feedback, writes down the lesson, and carries it into the next attempt. The model itself does not need to be retrained.

Work Automation stretched that idea across an entire software team.

A plan was proposed, challenged, revised, implemented, and then judged again by validation, CI, and the post-mortem.

The agents did not always agree, and that was the point.

## The Pipeline That Reviewed the Wrong Code

Dogfooding this thing on myself produced some genuinely funny failures.

In one pipeline, the verifier returned a blocking issue and two warnings. The review was detailed, specific, and completely wrong.

It had reviewed my host checkout instead of the pipeline's isolated branch. The two copies contained different versions of the same file.

The agent had done an excellent job reviewing the wrong code.

After that, every verifier had to prove which branch and exact commit it was looking at before it could review anything. If the branch did not match, it stopped. No fallback, no guessing, and no trying to be helpful.

It is the kind of bug that makes a demo look brilliant and a real system impossible to trust.

## The Agent That Worked for 30 Days

Another pipeline finished its build watch and knew the final outcome. Before saving that outcome, it tried to capture one last network snapshot. The process died during that tiny gap.

The database continued reporting that the pipeline was `running` for roughly 30 days.

It was not doing a month-long chain of thought. It was dead.

I changed the order immediately: save the final status first, collect any optional evidence second, and clean up last.

Other runs found similarly unexciting but important bugs:

- A missing report could be mistaken for a completed stage.
- A build watcher could return a false green while a merge conflict remained.
- A resumed pipeline could reuse an old pull request description.
- A restarted process could look like a genuine plan revision.
- An agent could test against a stale branch.

I learned pretty fast that a smarter model cannot save you from a dishonest control plane. Hand it the wrong code or lose its answer, and no amount of intelligence fixes that.

## Learning from Every Run

The post-mortem agent began as a way to summarize what happened. It eventually became the most important part of the project.

After every pipeline, it inspected the reports, logs, retries, failures, and final result. It looked for contradictions and strange behavior. It also looked for knowledge worth keeping.

If an agent tried three test commands before finding the one that worked, the successful command could become a reusable repository skill. If the researcher discovered an important architecture fact, it could be added to the repository's codebase map. If a failure exposed a flaw in the pipeline itself, the post-mortem could propose a framework fix.

Future agents read those skills and maps before starting their work.

That created a recursive loop:

![The Work Automation learning loop runs the pipeline, observes failures, records lessons, requires human approval, and gives approved memory to the next run](/assets/images/work-automation/learning-loop.svg)

The wrong-code review became a mandatory branch check. The 30-day zombie pipeline changed the finalization order. Repeated build failures became documented recipes.

The system was not retraining its models or autonomously rewriting itself. Every lasting change still required human approval. But each pipeline could leave the next one with better instructions and better memory.

That was the part I found most exciting.

By the end of the summer, those pieces had become a real system rather than a stack of experiments.

![The final Work Automation system runs multiple isolated single- or multi-repository pipelines concurrently, uses Claude, GPT, Codex, and Gemini as independent reviewers, tracks durable state, and adds human-approved lessons to shared repository memory](/assets/images/work-automation/final-system.svg){: .architecture-diagram }

## Harper Turns It into an Experiment

In May, I took on a summer intern, Harper Austin.

Until then, Work Automation was heavily shaped around my laptop, my repositories, my credentials, and my habits. Harper worked on separating the reusable framework from all the assumptions I had stopped noticing.

A project does not become reusable just because the README says it is.

Harper also explored how we could host the system at a larger scale. We investigated a managed hosted-agent platform, which forced us to think about identity, credentials, repository cloning, persistent workspaces, and long-running jobs.

The platform was not mature enough for the version of the workflow we were trying to run. We ultimately deployed it to a provisioned virtual machine where we controlled the filesystem, tools, processes, and container runtime.

A provisioned VM is not the exciting ending on an architecture slide. It worked, which mattered more.

More importantly, Harper tested whether this elaborate system actually helped.

To keep the comparison fair, he stayed within one model family—GPT-5.6 Sol, Terra, and Luna—and varied how the agents worked together.

Then he put them through SWE-bench Pro, a grueling exam made from difficult software-engineering tasks in real repositories. The question was simple: did all this specialization, arguing, and revising actually beat sending in one agent alone?

In our internal experiment, the best tested Work Automation configuration improved performance by more than 10% over its single-agent comparison.

That was the validation I wanted. It suggested that the workflow itself—the specialized roles, independent review, and back-and-forth—was not just theater. It was actually doing something.

I am not going to pretend one internal study proves anything about every model and every codebase. Harper's research deliberately stayed narrow. [Other researchers](https://arxiv.org/abs/2509.23537) have found similar benefits from mixing model families on general-reasoning tests, although those were not coding benchmarks.

That lines up with what I observed: Claude, GPT, Codex, and Gemini did not make identical mistakes. Their differences were useful.

## Then GitHub Announced HydraFusion

On September 4, GitHub introduced Project HydraFusion.

HydraFusion chooses whether to let one model work alone, let an efficient model try before escalating to a stronger one, or have one model produce an answer and another independently critique it.

What caught my attention was not simply that GitHub was using multiple models. It was everything around them: isolated critics, bounded execution, quality gates, safe failure behavior, routing rules, and accounting for every step.

Those details had consumed most of my summer.

HydraFusion and Work Automation still solve different problems. HydraFusion chooses how models should cooperate inside a coding request. Work Automation manages the longer journey around that request: repositories, isolated environments, tests, pull requests, CI, human checkpoints, and memory from earlier tasks.

I could imagine the two ideas fitting together. HydraFusion could eventually handle the reasoning inside one Work Automation stage while the outer factory manages the full engineering lifecycle.

The biggest thing I would borrow from HydraFusion is adaptive routing. Work Automation still follows rules I configured by hand. I want the next version to learn which tasks truly benefit from four reviewers, which dissenting reviews predict later failures, and when a cheaper model would have been enough.

## The Summer by the Numbers

- **167 commits** from April through August
- **30 days** that one very dead pipeline claimed to be running
- **More than 10% improvement** from the best tested configuration in our internal SWE-bench Pro experiment
- **1 provisioned virtual machine** that was much less exciting than a hosted-agent architecture and much more useful at the time

## What I Would Keep

For anyone building an orchestration workflow, I would start much smaller than Work Automation became.

Give research, implementation, and validation to separate roles. Save their handoffs somewhere outside the chat session. Keep validation read-only. Give implementation an isolated workspace. Make failures explicit and recoverable.

Only then would I add multi-model review, post-mortems, and human-approved repository memory.

The multi-model arguments are the fun part. Honest state management is what makes any of it trustworthy.

## Acknowledgments

Work Automation was collaborative from the beginning.

- **[Johnny Leek](https://www.linkedin.com/in/johnny-leek/)** gave me the initial working orchestration template that became the project's starting point.
- **[Harper Austin](https://www.linkedin.com/in/harper-austin-523743276/)**, the summer intern I mentored, helped abstract the framework, explored scalable serving patterns, moved the system toward a practical deployment, and ran the internal benchmark study.
- **[John Vicondoa](https://www.linkedin.com/in/jvicondoa/)**, Principal Software Engineer, built the Docker-in-Docker system that made isolated concurrent development environments possible and helped shape the broader software-factory ideas behind the project.
- **[Salman Quazi](https://www.linkedin.com/in/salmanquazi/)**, my former Director of Engineering, and **[Lucy Ulanova](https://www.linkedin.com/in/liudmilaulanova/)**, my former manager and Harper's manager during the internship, gave us the support and space to explore the work. I am extremely grateful to both of them for trusting me and for having the vision to give me room to help lead some of our agentic bets.

## What's Next

Work Automation is still a work in progress. It continues to find new ways to break, although the failures are at least more interesting now.

The next step is to make model routing less dependent on rules I wrote by hand. I want the system to learn from its own history: which agents were useful, which reviews predicted real bugs, which stages cost the most, and when a simpler workflow would have worked just as well.

I started this project because I was tired of manually coordinating coding agents. By the end of the summer, we had built a team of agents that could coordinate one another, argue about the work, remember what they learned, and improve the process for the next assignment.

That is still the part I am most proud of.
