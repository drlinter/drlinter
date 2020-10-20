import xml.etree.ElementTree as ET


def tagEditor(tag):
    return tag.replace("{http://www.gupro.de/GXL/gxl-1.0.dtd}","")



def faultCodeTranslator(fault):
    if fault == "f01":
        return "Wrong initialization."
    elif fault == "f02":
        return "Missing stepping the environment."
    elif fault == "f03":
        return "Missing terminal state."
    elif fault == "f04":
        return "Missing reset/close environment."
    elif fault == "f05":
        return "Missing exploration."
    elif fault == "f06":
        return "Wrong update rule."
    elif fault == "f07":
        return "Suboptimal exploration rate."
    elif fault == "f08":
        return "Suboptimal network update frequency."
    elif fault == "f09":
        return "Wrong network update."
    elif fault == "f10":
        return "Wrong activation for output"
    elif fault == "f11":
        return "Wrong output."

    else:
        return None


def main(inputGraphName):
    output = ""
    faults = []
    buggyNodes = []

    grooveOutputName = inputGraphName

    tree = ET.parse(f'graphs/{grooveOutputName}.gst')
    root = tree.getroot()


    for edge in root.iter("{http://www.gupro.de/GXL/gxl-1.0.dtd}edge"):
        fromNode = edge.attrib.get("from")
        toNode = edge.attrib.get("to")
        flag = False
        if fromNode == toNode and fromNode in buggyNodes:
            for strTag in edge.iter("{http://www.gupro.de/GXL/gxl-1.0.dtd}string"):
                flagAndFaultCode = strTag.text
                flag, faultCode = flagAndFaultCode.split(":")
                faults.append(faultCode)

        if fromNode == toNode:
            for strTag in edge.iter("{http://www.gupro.de/GXL/gxl-1.0.dtd}string"):
                if strTag.text == "type:Faults":
                    buggyNodes.append(fromNode)


    if not faults:
        output = "There is no identified fault in the DNN script"
    else:
        for fault in faults:
            output += faultCodeTranslator(fault)+"\n\r"


    return output

if __name__ == "__main__":

    print(main("grooveOut_01"))

