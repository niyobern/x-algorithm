# X For You Feed Algorithm

This repository contains the core recommendation system powering the "For You" feed on X. It combines in-network content (from accounts you follow) with out-of-network content (discovered through ML-based retrieval) and ranks everything using a Grok-based transformer model.

> **Note:** The transformer implementation is ported from the [Grok-1 open source release](https://github.com/xai-org/grok-1) by xAI, adapted for recommendation system use cases.

## Table of Contents

- [Updates — May 15th, 2026](#updates--may-15th-2026)
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Components](#components)
  - [Home Mixer](#home-mixer)
  - [Thunder](#thunder)
  - [Phoenix](#phoenix)
  - [Candidate Pipeline](#candidate-pipeline)
- [How It Works](#how-it-works)
  - [Pipeline Stages](#pipeline-stages)
  - [Scoring and Ranking](#scoring-and-ranking)
  - [Filtering](#filtering)
- [Key Design Decisions](#key-design-decisions)
- [30-Day X Growth System](#30-day-x-growth-system)
- [License](#license)

---

## Updates — May 15th, 2026

This release updates the For You algorithm code, including a runnable end-to-end inference pipeline alongside new components for content understanding, ads, and candidate sourcing.

1. **End-to-end inference pipeline:** A new [`phoenix/run_pipeline.py`](phoenix/run_pipeline.py) replaces the separate `run_ranker.py` and `run_retrieval.py` scripts with a single entry point that runs **retrieval → ranking** from exported checkpoints, mirroring how the two stages are composed in production.

2. **Pre-trained model artifacts:** A pre-trained mini Phoenix model (256-dim embeddings, 4 attention heads, 2 transformer layers) is now packaged as a ~3 GB archive distributed via Git LFS, enabling out-of-the-box inference without training your own model first.

3. **Grox content-understanding pipeline:** A new [`grox/`](grox/) service is included, providing classifiers, embedders, and a task-execution engine for content understanding workloads such as spam detection, post-category classification, and PTOS policy enforcement.

4. **Ads blending system:** Includes a new [`home-mixer/ads/`](home-mixer/ads/) module that handles ad injection and positioning within the feed, including brand-safety tracking that respects sensitive content boundaries.

5. **Query hydrators:** Home mixer now hydrates user context including followed topics, starter packs, impression bloom filters, IP, mutual follow graphs, and served history.

6. **Candidate hydrators:** Additional hydrators for engagement counts, brand safety signals, language codes, media detection, quote post expansion, mutual follow scores, and more.

7. **Candidate sources:** Adds sources for ads, who to follow, Phoenix MoE, Phoenix topics, prompts, and updates Thunder/Phoenix ones.

---

## Overview

The For You feed algorithm retrieves, ranks, and filters posts from two sources:

1. **In-Network (Thunder)**: Posts from accounts you follow
2. **Out-of-Network (Phoenix Retrieval)**: Posts discovered from a global corpus

Both sources are combined and ranked together using **Phoenix**, a Grok-based transformer model that predicts engagement probabilities for each post. The final score is a weighted combination of these predicted engagements.

We have eliminated every single hand-engineered feature and most heuristics from the system. The Grok-based transformer does all the heavy lifting by understanding your engagement history (what you liked, replied to, shared, etc.) and using that to determine what content is relevant to you.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    FOR YOU FEED REQUEST                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         HOME MIXER                                          │
│                                    (Orchestration Layer)                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                   QUERY HYDRATION                                   │   │
│   │  ┌──────────────────────────┐    ┌──────────────────────────────────────────────┐   │   │
│   │  │ User Action Sequence     │    │ User Features                                │   │   │
│   │  │ (engagement history)     │    │ (following list, preferences, etc.)          │   │   │
│   │  └──────────────────────────┘    └──────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                  CANDIDATE SOURCES                                  │   │
│   │         ┌─────────────────────────────┐    ┌────────────────────────────────┐       │   │
│   │         │        THUNDER              │    │     PHOENIX RETRIEVAL          │       │   │
│   │         │    (In-Network Posts)       │    │   (Out-of-Network Posts)       │       │   │
│   │         │                             │    │                                │       │   │
│   │         │  Posts from accounts        │    │  ML-based similarity search    │       │   │
│   │         │  you follow                 │    │  across global corpus          │       │   │
│   │         └─────────────────────────────┘    └────────────────────────────────┘       │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                      HYDRATION                                      │   │
│   │  Fetch additional data: core post metadata, author info, media entities, etc.       │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                      FILTERING                                      │   │
│   │  Remove: duplicates, old posts, self-posts, blocked authors, muted keywords, etc.   │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                       SCORING                                       │   │
│   │  ┌──────────────────────────┐                                                       │   │
│   │  │  Phoenix Scorer          │    Grok-based Transformer predicts:                   │   │
│   │  │  (ML Predictions)        │    P(like), P(reply), P(repost), P(click)...          │   │
│   │  └──────────────────────────┘                                                       │   │
│   │               │                                                                     │   │
│   │               ▼                                                                     │   │
│   │  ┌──────────────────────────┐                                                       │   │
│   │  │  Weighted Scorer         │    Weighted Score = Σ (weight × P(action))            │   │
│   │  │  (Combine predictions)   │                                                       │   │
│   │  └──────────────────────────┘                                                       │   │
│   │               │                                                                     │   │
│   │               ▼                                                                     │   │
│   │  ┌──────────────────────────┐                                                       │   │
│   │  │  Author Diversity        │    Attenuate repeated author scores                   │   │
│   │  │  Scorer                  │    to ensure feed diversity                           │   │
│   │  └──────────────────────────┘                                                       │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                      SELECTION                                      │   │
│   │                    Sort by final score, select top K candidates                     │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              FILTERING (Post-Selection)                             │   │
│   │                 Visibility filtering (deleted/spam/violence/gore etc)               │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     RANKED FEED RESPONSE                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Home Mixer

**Location:** [`home-mixer/`](home-mixer/)

The orchestration layer that assembles the For You feed. It leverages the `CandidatePipeline` framework with the following stages:

| Stage | Description |
|-------|-------------|
| Query Hydrators | Fetch user context (engagement history, following list) |
| Sources | Retrieve candidates from Thunder and Phoenix |
| Hydrators | Enrich candidates with additional data |
| Filters | Remove ineligible candidates |
| Scorers | Predict engagement and compute final scores |
| Selector | Sort by score and select top K |
| Post-Selection Filters | Final visibility and dedup checks |
| Side Effects | Cache request info for future use |

The server exposes a gRPC endpoint (`ScoredPostsService`) that returns ranked posts for a given user.

---

### Thunder

**Location:** [`thunder/`](thunder/)

An in-memory post store and realtime ingestion pipeline that tracks recent posts from all users. It:

- Consumes post create/delete events from Kafka
- Maintains per-user stores for original posts, replies/reposts, and video posts
- Serves "in-network" post candidates from accounts the requesting user follows
- Automatically trims posts older than the retention period

Thunder enables sub-millisecond lookups for in-network content without hitting an external database.

---

### Phoenix

**Location:** [`phoenix/`](phoenix/)

The ML component with two main functions:

#### 1. Retrieval (Two-Tower Model)
Finds relevant out-of-network posts:
- **User Tower**: Encodes user features and engagement history into an embedding
- **Candidate Tower**: Encodes all posts into embeddings
- **Similarity Search**: Retrieves top-K posts via dot product similarity

#### 2. Ranking (Transformer with Candidate Isolation)
Predicts engagement probabilities for each candidate:
- Takes user context (engagement history) and candidate posts as input
- Uses special attention masking so candidates cannot attend to each other
- Outputs probabilities for each action type (like, reply, repost, click, etc.)

See [`phoenix/README.md`](phoenix/README.md) for detailed architecture documentation.

---

### Candidate Pipeline

**Location:** [`candidate-pipeline/`](candidate-pipeline/)

A reusable framework for building recommendation pipelines. Defines traits for:

| Trait | Purpose |
|-------|---------|
| `Source` | Fetch candidates from a data source |
| `Hydrator` | Enrich candidates with additional features |
| `Filter` | Remove candidates that shouldn't be shown |
| `Scorer` | Compute scores for ranking |
| `Selector` | Sort and select top candidates |
| `SideEffect` | Run async side effects (caching, logging) |

The framework runs sources and hydrators in parallel where possible, with configurable error handling and logging.

---

## How It Works

### Pipeline Stages

1. **Query Hydration**: Fetch the user's recent engagements history and metadata (eg. following list)

2. **Candidate Sourcing**: Retrieve candidates from:
   - **Thunder**: Recent posts from followed accounts (in-network)
   - **Phoenix Retrieval**: ML-discovered posts from the global corpus (out-of-network)

3. **Candidate Hydration**: Enrich candidates with:
   - Core post data (text, media, etc.)
   - Author information (username, verification status)
   - Video duration (for video posts)
   - Subscription status

4. **Pre-Scoring Filters**: Remove posts that are:
   - Duplicates
   - Too old
   - From the viewer themselves
   - From blocked/muted accounts
   - Containing muted keywords
   - Previously seen or recently served
   - Ineligible subscription content

5. **Scoring**: Apply multiple scorers sequentially:
   - **Phoenix Scorer**: Get ML predictions from the Phoenix transformer model
   - **Weighted Scorer**: Combine predictions into a final relevance score
   - **Author Diversity Scorer**: Attenuate repeated author scores for diversity
   - **OON Scorer**: Adjust scores for out-of-network content

6. **Selection**: Sort by score and select the top K candidates

7. **Post-Selection Processing**: Final validation of post candidates to be served

---

### Scoring and Ranking

The Phoenix Grok-based transformer model predicts probabilities for multiple engagement types:

```
Predictions:
├── P(favorite)
├── P(reply)
├── P(repost)
├── P(quote)
├── P(click)
├── P(profile_click)
├── P(video_view)
├── P(photo_expand)
├── P(share)
├── P(dwell)
├── P(follow_author)
├── P(not_interested)
├── P(block_author)
├── P(mute_author)
└── P(report)
```

The **Weighted Scorer** combines these into a final score:

```
Final Score = Σ (weight_i × P(action_i))
```

Positive actions (like, repost, share) have positive weights. Negative actions (block, mute, report) have negative weights, pushing down content the user would likely dislike.

---

### Filtering

Filters run at two stages:

**Pre-Scoring Filters:**
| Filter | Purpose |
|--------|---------|
| `DropDuplicatesFilter` | Remove duplicate post IDs |
| `CoreDataHydrationFilter` | Remove posts that failed to hydrate core metadata |
| `AgeFilter` | Remove posts older than threshold |
| `SelfpostFilter` | Remove user's own posts |
| `RepostDeduplicationFilter` | Dedupe reposts of same content |
| `IneligibleSubscriptionFilter` | Remove paywalled content user can't access |
| `PreviouslySeenPostsFilter` | Remove posts user has already seen |
| `PreviouslyServedPostsFilter` | Remove posts already served in session |
| `MutedKeywordFilter` | Remove posts with user's muted keywords |
| `AuthorSocialgraphFilter` | Remove posts from blocked/muted authors |

**Post-Selection Filters:**
| Filter | Purpose |
|--------|---------|
| `VFFilter` | Remove posts that are deleted/spam/violence/gore etc. |
| `DedupConversationFilter` | Deduplicate multiple branches of the same conversation thread |

---

## Key Design Decisions

### 1. No Hand-Engineered Features
The system relies entirely on the Grok-based transformer to learn relevance from user engagement sequences. No manual feature engineering for content relevance. This significantly reduces the complexity in our data pipelines and serving infrastructure.

### 2. Candidate Isolation in Ranking
During transformer inference, candidates cannot attend to each other—only to the user context. This ensures the score for a post doesn't depend on which other posts are in the batch, making scores consistent and cacheable.

### 3. Hash-Based Embeddings
Both retrieval and ranking use multiple hash functions for embedding lookup

### 4. Multi-Action Prediction
Rather than predicting a single "relevance" score, the model predicts probabilities for many actions.

### 5. Composable Pipeline Architecture
The `candidate-pipeline` crate provides a flexible framework for building recommendation pipelines with:
- Separation of pipeline execution and monitoring from business logic
- Parallel execution of independent stages and graceful error handling
- Easy addition of new sources, hydrations, filters, and scorers

---

## 30-Day X Growth System

This section implements a practical 30-day growth operating system for creators who want consistent audience growth on X. It is included as an applied playbook that mirrors how this project emphasizes systematic iteration, measurement, and optimization.

### 1) Set Your Lane

- Pick 1 audience
- Pick 1 theme
- Pick 1 repeatable promise

Formula:

`I help [audience] get [result] with [topic].`

### 2) Use 4 Content Pillars

1. **Education** — Teach something useful
2. **Opinion** — Share a strong take
3. **Proof** — Show results, case studies, lessons
4. **Personal** — Story, struggle, belief, behind the scenes

### 3) Weekly Posting Rhythm

- **Mon:** Actionable tip/thread
- **Tue:** Hot take/opinion
- **Wed:** Case study or breakdown
- **Thu:** Personal story with lesson
- **Fri:** Contrarian insight
- **Sat:** Short list post or quick win
- **Sun:** Recap + question/opinion bait

### 4) Daily Execution Rule

- Post 1–3 times per day
- Spend at least 30 minutes engaging before posting
- Spend at least 60 minutes replying after posting
- Save every strong hook and reuse winners

### 5) 30 Daily Prompts

1. Biggest mistake beginners make in your niche
2. Unpopular opinion you strongly believe
3. 3-step framework for one small result
4. What you would do from zero today
5. Myth vs reality in your niche
6. Personal lesson from failure
7. Weekly recap of what worked
8. “Stop doing this” post
9. Tool/resource recommendation
10. Mini case study
11. Before/after transformation
12. Answer a common objection
13. List of beginner traps
14. Bold prediction
15. Your process in 5 steps
16. Breakdown of a successful creator/post
17. Share a small win with lesson
18. “If I had to start over” post
19. One habit that changed results
20. Misconception people repeat
21. Weekly recap + audience question
22. Thread with practical checklist
23. Story from your own journey
24. Comparison post: good vs bad approach
25. Common advice you disagree with
26. Post a template people can copy
27. Share data/result and what it means
28. “Do this instead” post
29. Best lesson from the month
30. Summary thread of top insights

### 6) Hook Templates

- Most people on X are doing ___ wrong.
- If you’re stuck at low views, start here:
- The harsh truth about growing on X:
- I’d do this if I had to go from 0 again.
- This small change can 10x your post performance.
- Unpopular opinion: ___
- Nobody tells creators this, but ___
- 3 mistakes killing your reach right now:
- Here’s the framework I use to ___
- Steal this template:

### 7) Post Structure

1. Hook
2. One core idea
3. 3–5 supporting points
4. Simple CTA

CTA examples:
- Agree or disagree?
- Want part 2?
- Which one is you?

### 8) KPI Tracker

Track each post for:

- Impressions
- Likes
- Replies
- Reposts
- Bookmarks
- Profile visits
- Follows gained
- Engagement rate
- Top hook used
- Content pillar used

### 9) Review Every 7 Days

- Identify top 20% posts
- Reuse their hooks
- Repackage the same idea in a new format
- Cut low-performing topics
- Keep themes that earn replies, saves, and follows

### 10) Goal

Don’t chase “millions” first. Optimize for:

- Better hooks
- Higher retention
- More shares
- More profile clicks

Viral reach usually comes from repeated strong packaging, not luck.

### Copy-Paste Weekly Calendar (4 Weeks)

Use this same pattern each week:

- **Monday:** Actionable tip/thread (Education)
- **Tuesday:** Strong opinion/hot take (Opinion)
- **Wednesday:** Case study/breakdown (Proof)
- **Thursday:** Story + lesson (Personal)
- **Friday:** Contrarian insight (Opinion)
- **Saturday:** Short list or quick win (Education)
- **Sunday:** Weekly recap + audience question (Proof/Opinion)

### 30 Ready-to-Post Example Prompts

1. Most people trying to grow on X are posting without a clear promise. Pick one.
2. Unpopular opinion: Consistency beats creativity when you’re under 10k followers.
3. Steal this 3-step framework to turn one idea into 7 posts.
4. If I had to start from 0 today, I’d focus on replies before threads.
5. Myth: You need viral posts to grow. Reality: You need repeatable clarity.
6. My biggest early mistake: posting insights with no clear CTA.
7. Week 1 recap: Here’s what performed best and why.
8. Stop writing clever hooks. Start writing specific hooks.
9. One tool I use to save winning hooks and reuse them every week.
10. Mini case study: One format change that doubled profile clicks.
11. Before: low engagement. After: one lane + one promise + steady growth.
12. “But my niche is too small.” Good. Small niches convert better.
13. 5 beginner traps that kill growth on X.
14. Prediction: Creator pages will look more like media brands in 12 months.
15. My 5-step content process from idea to post.
16. Breakdown: Why this creator’s thread worked (and what to copy).
17. Small win: one post brought qualified followers, not just impressions.
18. If I restarted today, I’d write for one person, not “everyone.”
19. One habit that changed everything: writing tomorrow’s hook today.
20. Misconception: More posts = more growth. Better framing matters more.
21. Weekly recap + question: Which format should I test next?
22. Checklist thread: 7 things to verify before posting.
23. Story: The post I almost didn’t publish — and what it taught me.
24. Good vs bad hook examples (and why one wins).
25. Common advice I disagree with: “Just be authentic.” Be useful first.
26. Copy this template for educational posts in any niche.
27. Here’s last week’s data and the one insight I’m keeping.
28. Do this instead: replace broad tips with specific, time-bound actions.
29. Best lesson from this month: clarity compounds.
30. Monthly summary thread: top insights, top posts, next experiments.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
