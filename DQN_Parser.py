import ast
import os

class RL_model:
    def __init__(self,values):
        self.parameters = values

    def connectionEdges(self, nodeId, nodeLabel):
        edges = ""
        if nodeLabel == "DRL-Program":
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('DQN')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>uses</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Environment')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>interacts-with</string>\n\t\t\t</attr>\n\t\t</edge>\n"

        elif nodeLabel == "DQN":
            dest_nodes = ["Hyperparameters", "Exploration", "target-network", "Q-network"]
            for node in dest_nodes:
                if node == "Exploration" and self.parameters.get("exploration_check")[1] != 'true':
                    continue
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get(node)}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>has</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:gamma = real:{self.parameters.get('gamma')[1]}</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:alpha = real:{self.parameters.get('learning_rate')[1]}</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:is_update_eq_valid = bool:{self.parameters.get('update_eq')[1]}</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:action_indication = bool:{self.parameters.get('output')[1]}</string>\n\t\t\t</attr>\n\t\t</edge>\n"

        elif nodeLabel == "Hyperparameters":
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:batchSize = int:{self.parameters.get('batch_size')[1]}</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:epochCount = int:{self.parameters.get('epoch_count')[1]}</string>\n\t\t\t</attr>\n\t\t</edge>\n"

        elif nodeLabel == "Exploration" and self.parameters.get("exploration_check")[1] == 'true':
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:decay_factor = real:{self.parameters.get('epsilon_decay')[1]}</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:explorationRate = real:{self.parameters.get('exploration_rate')[1]}</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:update_exploration_rate = bool:{self.parameters.get('update_exploration_rate')[1]}</string>\n\t\t\t</attr>\n\t\t</edge>\n"

        elif nodeLabel == "target-network":
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Q-network')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>Q-values_to_train</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:updateFrequency = int:{self.parameters.get('update_freq')[1]}</string>\n\t\t\t</attr>\n\t\t</edge>\n"

        elif nodeLabel == "Q-network":
            if self.parameters.get('update_target')[1] == 'true':
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('target-network')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>periodically_updates</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            if self.parameters.get("in_correct_step")[1] == 'true':
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Step')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>send_action</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:last_layer_activation = \"{self.parameters.get('net_last_layer')[1]}\"</string>\n\t\t\t</attr>\n\t\t</edge>\n"

        elif nodeLabel == "Environment":
            if self.parameters.get('initialize_env_correct')[1] == 'true':
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Initialize')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>starts-by</string>\n\t\t\t</attr>\n\t\t</edge>\n"
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>let:name = \"{self.parameters.get('initialize_env')[1]}\"</string>\n\t\t\t</attr>\n\t\t</edge>\n"

        elif nodeLabel == "Initialize":
            if self.parameters.get('initialize_env_correct')[1] == 'true' and self.parameters.get("in_correct_step")[
                1] == 'true':
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Step')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>continues-by</string>\n\t\t\t</attr>\n\t\t</edge>\n"

        elif nodeLabel == "Step":
            if self.parameters.get("in_correct_step")[1] == 'true':
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>repeats</string>\n\t\t\t</attr>\n\t\t\t<attr name=\"layout\">\n\t\t\t\t<string>500 -9 927 479 1025 478 927 479 11</string>\n\t\t\t</attr>\n\t\t</edge>\n"
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Q-network')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>receive_state</string>\n\t\t\t</attr>\n\t\t</edge>\n"
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Terminalstate')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>detect</string>\n\t\t\t</attr>\n\t\t</edge>\n"

        elif nodeLabel == "Terminalstate":
            if self.parameters.get('terminate_isCorrect')[1] == 'true' and self.parameters.get("in_correct_step")[
                1] == 'true':
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Step')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>reset</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            if self.parameters.get('env_close')[1] == 'true' and self.parameters.get('initialize_env_correct')[1] == 'true' :
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Initialize')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>close</string>\n\t\t\t</attr>\n\t\t</edge>\n"


        return edges

    def createNodesAndEdges(self):
        nodes = ["DRL-Program", "DQN", "Hyperparameters","Exploration", "target-network"
                 ,"Q-network", "Environment", "Initialize", "Step", "Terminalstate"]
        nodes_coordination = {0:[108 ,93 ,83,18],1:[443 ,77 ,147 ,72],2:[429, 243, 122 ,54],3:[260, 258, 133, 54],4:[581, 288, 141, 54], 5:[837, 94, 65, 18],6:[128, 518, 126, 36], 7:[523, 525, 53 ,18], 8:[809, 512, 95, 36],9:[789, 604, 116, 72]}
        self.graphNodes = {}
        self.RL_nodes = ""
        self.RL_edges = ""
        nodeCounter  = 0

        # adding nodes to the graph
        while nodeCounter<len(nodes):
            if nodes[nodeCounter] == "Exploration" and self.parameters.get("exploration_check")[1] != 'true':
                nodeCounter += 1
                continue
            if nodes[nodeCounter] == "Initialize" and self.parameters.get('initialize_env_correct')[1] != 'true':
                nodeCounter += 1
                continue
            if nodes[nodeCounter] == "Step" and self.parameters.get("in_correct_step")[1] != 'true':
                nodeCounter += 1
                continue
            nodeCord = nodes_coordination.get(nodeCounter)
            self.RL_nodes += f"\t\t<node id=\"n{nodeCounter}\">\n"
            self.RL_nodes += f"\t\t\t<attr name=\"layout\">\n\t\t\t\t<string>{nodeCord[0]} {nodeCord[1]} {nodeCord[2]} {nodeCord[3]}</string>\n\t\t\t</attr>\n"
            self.RL_nodes += "\t\t</node>\n"
            self.graphNodes[nodes[nodeCounter]] = f"n{nodeCounter}"
            nodeCounter += 1


        # adding nodes' labels and edges amongst the nodes
        nodeCounter = 0
        while nodeCounter<len(nodes):
            nodeId = f"n{nodeCounter}"
            nodeLabel = nodes[nodeCounter]
            if nodeLabel == "Exploration" and self.parameters.get("exploration_check")[1] != 'true':
                nodeCounter += 1
                continue
            if nodes[nodeCounter] == "Initialize" and self.parameters.get('initialize_env_correct')[1] != 'true':
                nodeCounter += 1
                continue
            if nodes[nodeCounter] == "Step" and self.parameters.get("in_correct_step")[1] != 'true':
                nodeCounter += 1
                continue

            self.RL_edges +=f"\t\t<edge from=\"{nodeId}\" to=\"{nodeId}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>type:{nodeLabel}</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            self.RL_edges += self.connectionEdges(nodeId, nodeLabel)
            nodeCounter += 1


    def creatModel(self, fileName):
        file = open(f'graphs/{fileName}.gst', 'w')
        file.write(
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<gxl xmlns=\"http://www.gupro.de/GXL/gxl-1.0.dtd\">\n")
        file.write("\t<graph role=\"graph\" edgeids=\"false\" edgemode=\"directed\" id=\"RL\">\n")
        file.write(self.RL_nodes)
        file.write(self.RL_edges)
        file.write("\t</graph>\n</gxl>")

        file.close()


graphValues = {"env_close":[None,"false"], "update_exploration_rate":[None, "false"],
               "terminal_state":[None,"false"], "initialize_env_correct":[None, "false"], "initialize_env":[None, None]
               , "exploration_check":[None, "false"], "net_last_layer":[None,None], "output":[None,"false"]
               ,"in_correct_step":[None, "false"], "update_eq":[None, "false"], "terminate_isCorrect":[None, "false"]}

def checkEnvReset(node):
    if isinstance(node, ast.Expr) and hasattr(node, "value") and hasattr(node.value, "func") and node.value.func.attr == "reset":
        return "true"
    else:
        return "false"

def findValueByVarName(varName):
    global graphValues
    for item in graphValues.values():
        if item[0] == varName:
            return item[1]
    return None

def checkexplorationUpdateEq(node):
    global graphValues

    if node.left.id != graphValues.get("exploration_rate")[0] and node.right.id != graphValues.get("exploration_rate")[0]:
        return "false"
    elif node.left.id == graphValues.get("exploration_rate")[0] and isinstance(node.op, ast.Mult) and findValueByVarName(node.right.id)<1:
        return "true"
    elif node.left.id == graphValues.get("exploration_rate")[0] and isinstance(node.op, ast.Div) and findValueByVarName(node.right.id)>1:
        return "true"
    elif node.right.id == graphValues.get("exploration_rate")[0] and isinstance(node.op, ast.Mult) and findValueByVarName(node.left.id)<1:
        return "true"
    elif node.right.id == graphValues.get("exploration_rate")[0] and isinstance(node.op, ast.Div) and findValueByVarName(node.left.id)>1:
        return "true"

    return "false"

def checkUpdateEq(node):
    if isinstance(node.left, ast.Name) and isinstance(node.right,ast.BinOp) and node.left.id == 'rewards':
        if node.right.left.attr =='gamma' and node.right.right.id == 'value_next':
            return True
    elif isinstance(node.right, ast.Name) and isinstance(node.left,ast.BinOp) and node.right.id == 'rewards':
        if node.left.left.attr =='gamma' and node.left.right.id == 'value_next':
            return True
    else:
        return False

def extractValues(root):
    global graphValues
    for node in ast.iter_child_nodes(root):
        # if hasattr(node, "lineno") and node.lineno == 53:
        #     print("test")

        #checking the e xploration
        if isinstance(node, ast.If) and hasattr(node, "test") and hasattr(node.test, "left") and isinstance(node.test.left, ast.Call) and hasattr(node.test.left.func, "attr") and node.test.left.func.attr == 'random' and hasattr(node.test, 'comparators') and node.test.comparators[0].id == 'epsilon':
            graphValues["exploration_check"] = [None, "true"]

        # checking the terminal state
        if isinstance(node, ast.If) and hasattr(node, "test") and hasattr(node.test, "id") and node.test.id == 'done':
            terminate_isCorrect = "false"
            for child in ast.iter_child_nodes(node):
                if checkEnvReset(child) == "true":
                    terminate_isCorrect = "true"
                    break
            graphValues["terminate_isCorrect"] = [None, terminate_isCorrect]

        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef) or isinstance(node,ast.For) or isinstance(node, ast.If) or isinstance(node, ast.While) or isinstance(node,ast.Compare):
            extractValues(node)

        #checking the correctness of target network update\
        if isinstance(node, ast.Expr) and hasattr(node, "value") and isinstance(node.value, ast.Call) and hasattr(node.value, "func") and hasattr(node.value.func, "attr") and node.value.func.attr == 'copy_weights' and node.value.func.value.id == 'TargetNet' and node.value.args[0].id=='TrainNet':
            graphValues["update_target"] = [None,'true']

        # extract gamma
        if isinstance(node,ast.Assign) and hasattr(node,"targets") and hasattr(node.targets[0], "id") and node.targets[0].id == 'gamma' and isinstance(node.value, ast.Num):
            graphValues["gamma"] = ["gamma", node.value.n]

        # extract update_freq
        if isinstance(node,ast.Assign) and hasattr(node,"targets") and hasattr(node.targets[0], "id") and node.targets[0].id == 'copy_step' and isinstance(node.value, ast.Num):
            graphValues["update_freq"] = ["copy_step", node.value.n]

        # extract batch size
        if isinstance(node,ast.Assign) and hasattr(node,"targets") and hasattr(node.targets[0], "id") and node.targets[0].id == 'batch_size' and isinstance(node.value, ast.Num):
            graphValues["batch_size"] = ["batch_size", node.value.n]

        # extract learning rate
        if isinstance(node, ast.Assign) and hasattr(node, "targets") and hasattr(node.targets[0], "id") and node.targets[0].id == 'lr' and isinstance(node.value, ast.Num):
            graphValues["learning_rate"] = ["lr", node.value.n]

        # extract epoch_count
        if isinstance(node, ast.Assign) and hasattr(node, "targets") and hasattr(node.targets[0], "id") and node.targets[0].id == 'N' and isinstance(node.value, ast.Num):
            graphValues["epoch_count"] = ["N", node.value.n]

        # extract exploration_rate
        if isinstance(node, ast.Assign) and hasattr(node, "targets") and hasattr(node.targets[0], "id") and \
                node.targets[0].id == 'epsilon' and isinstance(node.value, ast.Num):
            graphValues["exploration_rate"] = ["epsilon", node.value.n]

        # extract epsilon_decay
        if isinstance(node, ast.Assign) and hasattr(node, "targets") and hasattr(node.targets[0], "id") and \
                node.targets[0].id == 'decay' and isinstance(node.value, ast.Num):
            graphValues["epsilon_decay"] = ["decay", node.value.n]

        #checking stepping environment
        if isinstance(node, ast.Assign) and hasattr(node, "value") and hasattr(node.value, "func") and hasattr(node.value.func, "attr") and node.value.func.attr == 'step':
            graphValues["in_correct_step"] = [None, "true"]

        # extract close environment
        if isinstance(node, ast.Expr) and hasattr(node, "value") and hasattr(node.value, "func") and hasattr(node.value.func, "attr") and node.value.func.attr == 'close' and hasattr(node.value.func, "value") and node.value.func.value.id == 'env':
            graphValues["env_close"] = [None, "true"]

        # extract environment initialization
        if isinstance(node, ast.Assign) and hasattr(node, "value") and hasattr(node.value, "func") and hasattr(node.value.func, "attr") and node.value.func.attr ==  'make':
            graphValues["initialize_env"] = [node.targets[0].id,node.value.args[0].s]
            graphValues["initialize_env_correct"] = [None, "true"]

        # check the correctness of output
        if isinstance(node, ast.Return) and hasattr(node,"value") and hasattr(node.value, "func") and hasattr(node.value.func,"attr") and node.value.func.attr == 'argmax':
            graphValues["output"] = [None, "true"]

        #extract network last layer
        if isinstance(node, ast.Assign) and hasattr(node, "targets") and hasattr(node.targets[0], "attr") and node.targets[0].attr == 'output_layer':
            for keyword in node.value.keywords:
                if keyword.arg == 'activation':
                    graphValues["net_last_layer"] = [node.targets[0].attr ,keyword.value.s]

        # extract update equation
        if isinstance(node, ast.Assign) and hasattr(node, "targets") and hasattr(node.targets[0], "id") and node.targets[0].id == 'actual_values':
            if node.value.func.attr == "where" and len(node.value.args) == 3 and node.value.args[0].id == "dones" and node.value.args[1].id == "rewards" and isinstance(node.value.args[2], ast.BinOp) and checkUpdateEq(node.value.args[2]):
                graphValues["update_eq"] = [node.targets[0].id, "true"]

        # checking the update_exploration_rate
        if isinstance(node, ast.Assign) and hasattr(node, "targets") and hasattr(node.targets[0], "id") and node.targets[0].id == 'epsilon' and hasattr(node.value, "func") and hasattr(node.value.func,"id") and node.value.func.id == 'max' and isinstance(node.value.args[1], ast.BinOp):
            graphValues.update({"update_exploration_rate": [None, checkexplorationUpdateEq(node.value.args[1])]})


def main(fileName):
    source = open(f"{fileName}", "r")
    FileNameWithoutPath = fileName.split(os.path.sep)[-1]
    file_name, file_extension = os.path.splitext(FileNameWithoutPath)
    tree = ast.parse(source.read())
    extractValues(tree)
    rl_model = RL_model(graphValues)
    rl_model.createNodesAndEdges()
    rl_model.creatModel(f"{file_name}")

if __name__ == '__main__':
    # main("buggy-clones/DQN_clean.py")
    # main("buggy-clones/DQN_Missing_close_reset_environment.py")
    # main("buggy-clones/DQN_Missing_exploration.py")
    # main("buggy-clones/DQN_missing_stepping.py")
    # main("buggy-clones/DQN_Missing_stepping_the_environment.py")
    # main("buggy-clones/DQN_Suboptimal_exploration_rate.py")
    # main("buggy-clones/DQN_Suboptimal_exploration_rate_2.py")
    # main("buggy-clones/DQN_Suboptimal_network_update_frequency.py")
    # main("buggy-clones/DQN_Wrong_activation_output.py")
    # main("buggy-clones/DQN_Wrong_initialization_missing.py")
    # main("buggy-clones/DQN_Wrong_output.py")
    # main("buggy-clones/DQN_Wrong_update_rule.py")
    main("buggy-clones/DQN_Missing_terminal_state.py")
