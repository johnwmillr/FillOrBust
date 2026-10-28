import pickle


fp = "./dice/trained_agents/trained_agent_1712728396_1000.pkl"

with open(fp, "rb") as fin:
    agent = pickle.load(fin)

sorted(agent.Q.table.values(), key=lambda x: max(x.values()), reverse=True)

table = agent.Q.table

{key: table[key] for key in sorted(table.keys(), key=lambda k: max(table[k].values()), reverse=True)[:5]}

print(agent)
