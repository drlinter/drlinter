import ast
import os

# def parse_line_code(line, annotation):
#     if annotation == "env":
#         if hasattr(line,'value') and hasattr(line.value,'value'):
#             return line.value.value
#     elif annotation == "gamma" :
#         if hasattr(line,'value') and hasattr(line.value,'value'):
#             return line.value.value
#     elif annotation == "lr" :
#         if hasattr(line,'value') and hasattr(line.value,'value'):
#             return line.value.value
#     elif annotation == "epoch":
#         if hasattr(line,'value') and hasattr(line.value,'value'):
#             return line.value.value

class RL_model:
    def __init__(self,values):
        self.parameters = values

    def connectionEdges(self, nodeId, nodeLabel):
        edges = ""
        if nodeLabel == "DRL-Program":
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('DQN')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>uses</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Environment')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>interacts-with</string>\n\t\t\t</attr>\n\t\t</edge>\n"

        elif nodeLabel == "DQN":
            dest_nodes = ["Hyperparameters", "Exploration",
                          # "target-network",
                          "Q-network"]
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
            if self.parameters.get('terminate_isCorrect')[1] == 'true' and self.parameters.get("in_correct_step")[1] == 'true':
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Step')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>reset</string>\n\t\t\t</attr>\n\t\t</edge>\n"
            if self.parameters.get('env_close')[1] == 'true' and self.parameters.get('initialize_env_correct')[1] == 'true' :
                edges += f"\t\t<edge from=\"{nodeId}\" to=\"{self.graphNodes.get('Initialize')}\">\n\t\t\t<attr name=\"label\">\n\t\t\t\t<string>close</string>\n\t\t\t</attr>\n\t\t</edge>\n"


        return edges

    def createNodesAndEdges(self):
        nodes = ["DRL-Program", "DQN", "Hyperparameters","Exploration"
             # , "target-network"
                 ,"Q-network", "Environment", "Initialize", "Step", "Terminalstate"]
        nodes_coordination = {0:[108 ,93 ,83,18],1:[443 ,77 ,147 ,72],2:[429, 243, 122 ,54],3:[260, 258, 133, 54],
                                # 4:[581, 288, 141, 54],
                              # 5:[837, 94, 65, 18],6:[128, 518, 126, 36], 7:[523, 525, 53 ,18], 8:[809, 512, 95, 36],9:[789, 604, 116, 72]}
                                4:[837, 94, 65, 18],5:[128, 518, 126, 36], 6:[523, 525, 53 ,18], 7:[809, 512, 95, 36],8:[789, 604, 116, 72]}
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

def parse_just_line_code(node, annotation):
    if node.body == []:
        line = ""
    else:
        line = node.body[0]

    if annotation in ["initialize_env", "gamma", "learning_rate", "epoch_count", "epsilon_decay", "alpha", "batch_size", "update_freq"]:
        return line.value.value
    elif annotation == "create_env":
        return line.value.args[0].value
    elif annotation in ["update_eq", "terminal_state", "in_correct_step", "exploration_check","initialize_env_correct", "update_exploration_rate", "env_close"]:
        return 'true'

def parse_multi_annotation(node, annotation):
    identified_annotations = {}
    annotations = annotation.split(sep=",")
    line = node.body[0]

    item_counter = -1
    for item in annotations:
        item_counter += 1
        if item == "_":
            continue
        elif item in ["in_correct_step", "env_close", "initialize_env_correct","terminate_isCorrect","terminal_state","exploration_check"]:
            identified_annotations[item] = [None,'true']
        else:
            identified_annotations[item] = [None, line.value.args[item_counter].value]

    return identified_annotations


class DRLinter:
    def __init__(self, source):
        self.file_name = source
        # self.model_info = {
        #     'exploration_check' : [None, 'false'],
        #     'update_exploration_rate'  : [None, 'false'],
        #     'update_target' : [None, 'true'],
        #     'decay' : [None, 0.9],                          #
        #     'epsilon_decay' : [None, 0.9],                        #
        #     'exploration_rate' : [None, 0.9],
        #     'action_indication' : [None, 'true'],
        #     'alpha' : [None, 0.9],                          #
        #     'gamma' : [None, 0.1],                          #
        #     'learning_rate' : [None, 0.01],                 #
        #     'update_eq' : [None, 'false'],
        #     'initialize_env'   : [None, 'CartPole-v0'],                #
        #     'initialize_env_correct' : [None, 'true'],
        #     'batch_size' : [None, 100],
        #     'epoch_count' : [None, 100],                    #
        #     'update_freq' : [None, 30],
        #     'net_last_layer' : [None, 'Nan'],
        #     'in_correct_step' : [None, 'true'],
        #     'terminate_isCorrect' : [None, 'false'],
        #     'env_close' : [None, 'false'],
        #     'output' : [None, 'true']
        # }

        self.model_info = {
            "env_close":[None,"false"],
            "update_exploration_rate":[None, "false"],
            "terminal_state":[None,"false"],
            "initialize_env_correct":[None, "false"],
            "initialize_env":[None, None],
            "exploration_check":[None, "false"],
            "net_last_layer":[None,"linear"],
            "output":[None,"true"],
            "in_correct_step":[None, "false"],
            "update_eq":[None, "true"],
            "terminate_isCorrect":[None, "false"],
            "update_target":[None, "false"],
            'gamma' : [None, 0.1],
            'update_freq' : [None, 30],
            'batch_size' : [None, 100],
            'learning_rate' : [None, 0.01],
            'epoch_count' : [None, 100],
            'exploration_rate' : [None, 0.9],
            'epsilon_decay' : [None, 0.9]
        }
        self.annotations = {}
        self.find_anottations()
        # self.parse_srcipt()


    def find_anottations(self):
        source_file = open(self.file_name,mode="r")
        lines = source_file.readlines()
        for line in lines:
            check_status = line.find("#@DRLinter")
            if check_status >=0:
                annotation = line[check_status+13:-1]
                line_number = lines.index(line)+1

                node = ast.parse(line.strip())

                if annotation.find(",")>0:
                    identified_annotations = parse_multi_annotation(node, annotation)
                    self.model_info.update(identified_annotations)
                    for item in identified_annotations.keys():
                        self.annotations[line_number] = item
                    continue
                else:
                    value = parse_just_line_code(node, annotation)


                if annotation == "create_env" :
                    annotation = "initialize_env"
                self.annotations[line_number] = annotation
                self.model_info[self.annotations[line_number]] = [None, value]

        # if self.model_info['terminal_state'][1] == 'true':
        #     self.model_info['terminate_isCorrect'] = [None, 'true']





    def parse_srcipt(self):
        source_file = open(file=self.file_name, mode="r")
        tree = ast.parse(source_file.read())
        for node in ast.walk(tree):
            if hasattr(node, "lineno") and node.lineno in self.annotations.keys():
                print(node.lineno)
                value = parse_just_line_code(node,self.annotations[node.lineno])
                self.model_info[self.annotations[node.lineno]] = value

def main(fileName):
    # FileNameWithoutPath = fileName.split(os.path.sep)[-1]
    FileNameWithoutPath = os.path.split(fileName)[-1]
    file_name, file_extension = os.path.splitext(FileNameWithoutPath)
    parser = DRLinter(fileName)
    parser.find_anottations()
    rl_model = RL_model(values=parser.model_info)
    rl_model.createNodesAndEdges()
    rl_model.creatModel(f"{file_name}")
    print(f"{fileName} is parsed....")




if __name__ == '__main__':
    main("buggy_so/47643678.py")
    # main("buggy_so/47750291.py")
    # main("buggy_so/49035549.py")
    # main("buggy_so/50308750.py")
    # main("buggy_so/51425688.py")
    # main("buggy_so/54385568.py")
