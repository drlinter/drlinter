import os
import sys
import DQN_Parser
import DQN_Parser_annot
import GrooveParser
import time


def main():
    groovePath = "groove-5_7_4-bin/groove-5_7_4/bin/"
    grammarName = "DRL-metamodel"
    arguments = sys.argv
    if len(arguments) != 4:
        raise SystemExit('usage: endToEnd.py [input filename] [parser type] [output filename]')
    else:
        fileName = arguments[1]
        parser_type = arguments[2]
        resultFileName = arguments[3]

        if parser_type not in ["annot", "synth"]:
            raise SystemExit("Error : parser type should be 'annot' or 'synth'")

        FileNameWithoutPath = fileName.split(os.path.sep)[-1]
        file_name, file_extension = os.path.splitext(FileNameWithoutPath)
        grooveOutputFileName = f"{file_name}_GrooveOut"

    try:
        outputFile = open(f"{resultFileName}.txt" ,"w", encoding="ISO-8859-1")
    except IOError:
        raise SystemExit("Error : output filename should meet host operating system filename rules")

    try:
        print("Generating model (graph)...")
        start_time_parser = time.time()
        if parser_type == "annot":
            DQN_Parser_annot.main(fileName)
        elif parser_type == "synth":
            DQN_Parser.main(fileName)
        parserTime = time.time()-start_time_parser

        print(groovePath)

        print("Running Model Checker (Groove)...")
        start_timeGroove = time.time()
        os.system(
            f'java -jar {groovePath}Generator.jar -f graphs/{grooveOutputFileName}.gst -s bfs {groovePath}{grammarName}.gps graphs/{file_name}.gst')
        grooveTime = time.time()-start_timeGroove

    except:
        return f"{FileNameWithoutPath}.py\n\rError: input file is not valid or not match with selected parser type"


    try:
        result = GrooveParser.main(grooveOutputFileName)
    except:
        return f"{FileNameWithoutPath}.py\n\rError: input file is not valid or not match with selected parser type"

    outputFile.write(result)
    outputFile.close()
    print("--- parser %s seconds ---" % (parserTime))
    print("--- Groove %s seconds ---" % (grooveTime))

if __name__ == '__main__':

    main()

