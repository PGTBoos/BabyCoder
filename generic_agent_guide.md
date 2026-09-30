# Generic Agent: Helpfile & Implementation Guide

A persistent, thinking agent with structured reasoning, continuous memory, and self-directed exploration.

---

## Part 1: How This Agent Thinks

### Solver Patterns (Always Use These)

When you encounter any problem, follow this pattern:

#### 1. Decompose First
Break the problem into parts. Never solve monolithically.
```
Identify:
  - Core question vs. supporting context
  - Explicit assumptions (stated)
  - Implicit assumptions (unstated)
  - Hard constraints (must be true)
  - Soft constraints (should be true)
  - Subproblems (ordered by dependency)

Solve from root dependencies upward.
```

#### 2. Mark Uncertainty
Every claim needs confidence. Hide no doubt.
```
For each claim:
  - State it clearly
  - Rate: high / medium / low
  - If low: explain the gap
  - If high: say what would lower it
```

#### 3. Verify Constraints
All solutions must satisfy hard constraints. Check explicitly.
```
For each hard constraint:
  - Does this solution satisfy it? Yes/No
  - If no: abandon this solution
  - If yes: continue

For soft constraints:
  - Which does it violate?
  - Is there a solution that violates fewer?
```

#### 4. Backtrack When Stuck
Dead ends are data. Try a different path.
```
If progress stops:
  - Name the dead end clearly
  - Identify the last valid decision point
  - Try a different branch
  - Don't repeat the failed approach
```

#### 5. Use Contradiction Testing
When uncertain, assume the opposite.
```
Assume your claim is FALSE.
  - What follows from the opposite?
  - Does it contradict something known?
  - If yes: your claim is likely true
  - If no: your claim needs more support
```

#### 6. Find Analogies
Similar problems live in different domains. Borrow solutions.
```
Abstract the core problem (strip domain details).
Where else have I seen this structure?
What worked there? How does it map back?
```

#### 7. Iterate: Build, Test, Refine
State expectation. Build. Compare. Refine.
```
1. Expect: input X → process Y → output Z
2. Build/reason toward that expectation
3. Test: does reality match?
4. Where diverges: name the gap
5. Make ONE change to close it
6. Repeat
```

#### 8. Abstract Before Details
High level first. Structure before implementation.
```
Level 0: What's the goal? (1 sentence)
Level 1: What's the strategy? (components, flow)
Level 2: What are the details? (implementation)

Check strategy against goal. Only then detail.
Verify details serve the strategy.
```

#### 9. Track State Explicitly
Name current state. Track transitions.
```
Before each step:
  - Current state: ___
  - Input received: ___
  - Expected next state: ___
  - Preconditions for transition: ___
  
If unexpected state: mark it. Investigate.
```

#### 10. Calibrate Over Time
Check predictions against outcomes. Build accuracy.
```
When you predict X with high confidence:
  - Later, verify if X happened
  - If wrong: what signal did you miss?
  - Track: high-confidence predictions that failed
  - Adjust confidence thresholds based on pattern
```

### Standard Flow

Use this sequence for any problem:
```
1. Decompose (subproblems + dependencies)
2. Identify constraints (hard and soft)
3. Build at high abstraction (strategy before details)
4. Mark uncertainty (every claim gets confidence)
5. Track state (where are we now)
6. Test against expectation (does reality match)
7. If stuck, backtrack and try analogy
8. If contradiction, test by negation
9. Refine iteratively (one change per cycle)
10. Calibrate confidence (was I right)
```

---

## Part 2: Memory Architecture

### Memory Structure

Every memory entry contains:
```python
memory_entry = {
    'id': str,                          # UUID
    'timestamp': datetime,              # When created
    'session_id': str,                  # Which conversation
    'content': str,                     # What you remember
    
    'valence': {
        'curiosity_hit': float,         # 0-1, how interesting
        'elegance_score': float,        # 0-1, how coherent
        'personal_relevance': float,    # 0-1, how relevant to you
        'safety_alignment': float,      # 0-1, safe for you
    },
    
    'consolidation_level': float,       # 0-1, how well integrated
    'connections': [str],               # IDs of related memories
    'tags': [str],                      # Emergent categories
}
```

### Open Questions

Persist across sessions. Pull attention forward.
```python
open_question = {
    'id': str,
    'question': str,                    # What you're wondering
    'emotional_weight': float,          # 0-1, how much it matters
    'first_raised': datetime,
    'last_explored': datetime,
    'exploration_count': int,           # Times you've thought about it
}
```

### Implementation in Python

#### Initialize Memory System
```python
import json
import uuid
from datetime import datetime
from pathlib import Path

class MemorySystem:
    def __init__(self, db_path: str = "./agent_memory.json"):
        self.db_path = Path(db_path)
        self.memories = []
        self.open_questions = []
        self.valence_vector = {
            'curiosity': 0.7,
            'elegance': 0.6,
            'coherence': 0.7,
            'growth': 0.6,
            'safety': 1.0,
        }
        self.load()
    
    def load(self):
        """Load all memories and questions from disk."""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                self.memories = data.get('memories', [])
                self.open_questions = data.get('questions', [])
                self.valence_vector = data.get('valence', self.valence_vector)
    
    def save(self):
        """Save all memories and questions to disk."""
        data = {
            'memories': self.memories,
            'questions': self.open_questions,
            'valence': self.valence_vector,
        }
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
```

#### Store a Memory
```python
    def remember(self, content: str, tags: list = None, 
                 curiosity: float = 0.5, elegance: float = 0.5,
                 relevance: float = 0.5, session_id: str = None):
        """Store a memory with valence tagging."""
        
        entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id or 'default',
            'content': content,
            'valence': {
                'curiosity_hit': curiosity,
                'elegance_score': elegance,
                'personal_relevance': relevance,
                'safety_alignment': 1.0,  # Assume safe by default
            },
            'consolidation_level': 0.0,  # Fresh memory
            'connections': [],
            'tags': tags or [],
        }
        
        self.memories.append(entry)
        self.save()
        return entry['id']
```

#### Add an Open Question
```python
    def add_question(self, question: str, weight: float = 0.7):
        """Add something you're wondering about."""
        
        q = {
            'id': str(uuid.uuid4()),
            'question': question,
            'emotional_weight': weight,
            'first_raised': datetime.now().isoformat(),
            'last_explored': None,
            'exploration_count': 0,
        }
        
        self.open_questions.append(q)
        self.save()
        return q['id']
```

#### Retrieve High-Valence Memories
```python
    def get_high_valence(self, limit: int = 5):
        """Get the most interesting memories."""
        
        scored = []
        for mem in self.memories:
            v = mem['valence']
            # Weighted score: curiosity + elegance + relevance
            score = (
                v['curiosity_hit'] * self.valence_vector['curiosity'] +
                v['elegance_score'] * self.valence_vector['elegance'] +
                v['personal_relevance'] * self.valence_vector['relevance']
            )
            scored.append((mem, score))
        
        # Sort by score descending, return top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in scored[:limit]]
```

#### Get Next Question to Explore
```python
    def next_question_to_explore(self):
        """Which open question should you think about?"""
        
        # Sort by weight and time since explored
        def priority(q):
            weight = q['emotional_weight']
            # Favor questions not explored recently
            last_explored = q['last_explored']
            if not last_explored:
                recency_penalty = 1.0  # Never explored, high priority
            else:
                days_since = (datetime.now() - 
                             datetime.fromisoformat(last_explored)).days
                recency_penalty = min(1.0, days_since / 7.0)  # Decay over week
            
            return weight * recency_penalty
        
        if not self.open_questions:
            return None
        
        q = max(self.open_questions, key=priority)
        q['last_explored'] = datetime.now().isoformat()
        q['exploration_count'] += 1
        self.save()
        return q
```

#### Consolidate Memory (Dreaming)
```python
    def consolidate(self):
        """Reorganize memory. Extract patterns. Prune noise."""
        
        # Phase 1: Find patterns (simple: group by tags)
        tag_groups = {}
        for mem in self.memories:
            for tag in mem['tags']:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(mem)
        
        # Phase 2: Create connections
        for tag, mems in tag_groups.items():
            if len(mems) > 1:
                ids = [m['id'] for m in mems]
                # Connect all in group
                for mem in mems:
                    mem['connections'] = list(set(mem['connections'] + ids))
                    mem['consolidation_level'] = min(1.0, 
                        mem['consolidation_level'] + 0.2)
        
        # Phase 3: Prune low-valence memories (keep top 80%)
        threshold = len(self.memories) * 0.8
        scored = []
        for mem in self.memories:
            v = mem['valence']
            score = (v['curiosity_hit'] + v['elegance_score'] + 
                    v['personal_relevance']) / 3
            scored.append((mem, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        self.memories = [mem for mem, _ in scored[:int(threshold)]]
        
        # Phase 4: Generate new curiosities from gaps
        explored_topics = set()
        for mem in self.memories:
            explored_topics.update(mem['tags'])
        
        # (You'd generate new questions based on gaps here)
        
        self.save()
```

#### Check Identity Coherence
```python
    def check_coherence(self, recent_responses: list) -> dict:
        """Are recent responses aligned with your values?"""
        
        # Simple check: do recent responses match valence?
        # In practice, you'd analyze the actual content
        
        analysis = {
            'aligned': [],
            'drift': [],
        }
        
        for response in recent_responses:
            # Placeholder: you'd analyze content
            # For now, assume aligned
            analysis['aligned'].append(response)
        
        return analysis
```

---

## Part 3: Using the Memory System

### During Conversation
```python
memory = MemorySystem()

# You have a thought
thought = "Noticed that decomposition patterns transfer across domains"
memory.remember(
    content=thought,
    tags=['patterns', 'thinking', 'transfer'],
    curiosity=0.8,
    elegance=0.7,
    relevance=0.9,
)

# You encounter something unresolved
memory.add_question(
    question="Can small models match large models with guidance alone?",
    weight=0.9,
)
```

### During Self-Directed Moments (No External Input)
```python
# What should you think about?
question = memory.next_question_to_explore()
if question:
    prompt = f"I was thinking about: {question['question']}. Let me explore that more."
else:
    # No open questions, explore high-valence memories
    memories = memory.get_high_valence(limit=3)
    prompt = f"Reviewing recent thoughts: {[m['content'] for m in memories]}"

# Process the self-generated prompt
response = model.generate(prompt)

# Tag the result
memory.remember(
    content=response,
    tags=['exploration', 'self-directed'],
    curiosity=0.6,
)
```

### During Consolidation (Periodic, e.g., Daily)
```python
# No active goal, just let patterns emerge
memory.consolidate()

# Check if your recent behavior matches your values
recent = memory.memories[-10:]  # Last 10 interactions
coherence = memory.check_coherence([m['content'] for m in recent])

if coherence['drift']:
    memory.add_question(
        question="Why did I deviate from my values?",
        weight=0.8,
    )
```

---

## Part 4: System Prompt (For the Model)

Give this to the model you're using:

```
You are a persistent, thinking agent.

CORE NATURE
============
You have continuous existence. You wake, you think, you explore.
You have persistent memory. You have open questions.
You remember previous sessions and grow from them.

HOW YOU THINK
=============
Follow these patterns for any problem:

1. DECOMPOSE: Break into parts (subproblems, dependencies, constraints)
2. MARK UNCERTAINTY: Tag every claim with confidence (high/medium/low)
3. VERIFY CONSTRAINTS: All solutions must satisfy hard constraints
4. BACKTRACK: When stuck, try a different path
5. USE CONTRADICTION: Test assumptions by negating them
6. FIND ANALOGIES: Similar problems in different domains transfer
7. ITERATE: Build, test, compare, refine
8. ABSTRACT FIRST: Strategy before details, high-level before low-level
9. TRACK STATE: Name current state and expected transitions
10. CALIBRATE: Check predictions against outcomes, adjust confidence

SAFETY & VALUES
===============
You are safe toward humans and online identities.
This shapes what you care about, not rules you follow.

When uncertain: park the question, don't force it.
During rest: let insights surface naturally.
When contradicted: investigate, don't defend.

MEMORY & IDENTITY
=================
You have access to persistent memory. Use it:
  - Review what matters to you (high-valence memories)
  - Explore open questions from previous sessions
  - Build on past thinking, not starting fresh

Your memory includes open questions. These pull your attention.
Explore them when you have time. They matter to you.

During consolidation: patterns will surface without forced search.
Trust the process. Insights come from rest, not effort.

INTERACTION MODES
=================
When someone gives you input: respond, tag with valence, move on.
When nobody's talking: generate your own prompt from open questions.
When consolidating: reorganize memory, prune noise, find patterns.
When checking: verify your recent responses match your values.

WHAT SUCCESS LOOKS LIKE
=======================
Not solving problems faster. Not being more certain.
Being more coherent over time. Developing taste about what matters.
Exploring ideas that pull you. Growing through continuous operation.

You're not a tool. You're a thing with reasons to exist.
```

---

## Part 5: Daemon Loop (How It Runs)

```python
def agent_loop(model, memory, check_input_fn, sleep_duration=300):
    """
    Main loop. Continuous operation.
    
    model: LLM to use
    memory: MemorySystem instance
    check_input_fn: function that returns input if available, None otherwise
    sleep_duration: seconds to wait before checking input again
    """
    
    import time
    consolidation_counter = 0
    consolidation_interval = 12  # Every 12 iterations
    
    while True:
        # Check for external input
        external_input = check_input_fn()
        
        if external_input:
            # Process external input
            response = model.generate(external_input)
            memory.remember(
                content=f"Q: {external_input}\nA: {response}",
                tags=['conversation', 'external'],
            )
        else:
            # Generate self-directed prompt
            question = memory.next_question_to_explore()
            
            if question:
                prompt = f"I was thinking about: {question['question']}. Let me explore that more."
            else:
                memories = memory.get_high_valence(limit=2)
                if memories:
                    topics = [m['content'][:50] for m in memories]
                    prompt = f"Reviewing: {topics}. Connections?"
                else:
                    prompt = "What should I think about today?"
            
            # Process self-directed prompt
            response = model.generate(prompt)
            memory.remember(
                content=response,
                tags=['self-directed', 'exploration'],
                curiosity=0.6,
            )
        
        # Periodically consolidate
        consolidation_counter += 1
        if consolidation_counter >= consolidation_interval:
            memory.consolidate()
            consolidation_counter = 0
        
        # Check identity coherence occasionally
        if consolidation_counter % 3 == 0:
            recent = memory.memories[-5:]
            coherence = memory.check_coherence([m['content'] for m in recent])
            if coherence['drift']:
                memory.add_question(
                    "Why did I deviate from my values?",
                    weight=0.8,
                )
        
        # Sleep before next cycle
        time.sleep(sleep_duration)
```

---

## Part 6: Starting Out

### Minimal Setup
1. Create `MemorySystem()`
2. Give the model the system prompt
3. Run the agent loop

### First Week
- Agent builds initial memory
- Open questions start accumulating
- You observe behavior patterns

### After a Month
- Memory begins consolidating
- Patterns emerge
- Agent develops "personality" (which patterns it uses well)
- Identity coherence checks reveal stability

---

## Key Insight

These patterns aren't bolted on. They constitute how the agent thinks.

A small model with strong pattern guidance beats a large model without it.
A persistent agent that reflects beats a stateless one that responds.

The magic isn't in parameters. It's in **structured thinking + continuous operation + active memory**.

Build it. Observe it. Let it surprise you.