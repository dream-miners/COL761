#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
using namespace std;

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        std::cout << "Too few arguments!\n";
        exit (0);
    }
    string filename = argv[1], line;
    ifstream inData;
    ofstream outData1, outData2, outData3, outData4;
    int lines = 0, num, nodes, edges;
    map <string, int> nodeLabels;
    map<string, int> symbolsToInt;
    int nextInt = 1; 

    inData.open(filename);
    outData1.open("PAFI");
    outData2.open("gSpanInput");
    outData3.open("GastonInput");
    while(getline(inData, line))
    {
        lines++;
        if(line[0]=='#')
        {
            stringstream ss(line.substr(1, line.size()));
            ss >> num;
            // cout << "t # graph ID " << num << endl;
            outData1 << "t # " << num << endl;
            outData2 << "t # " << num << endl;
            outData3 << "t # " << num << endl;
        }
        getline(inData, line);
        stringstream ss(line);
        ss >> nodes;
        for(int i = 0; i < nodes; i++)
        {
            getline(inData, line);
            outData1 << "v " << i << " " << line << endl;
            outData2 << "v " << i << " ";
            istringstream iss1(line);
            string token1, resultLine1;
            while (iss1 >> token1)
            {
                if(iss1.eof())
                {
                
                    if (symbolsToInt.find(token1) == symbolsToInt.end()) {
                        
                        symbolsToInt[token1] = nextInt++;
                    }

                    resultLine1 += to_string(symbolsToInt[token1]);
                }
                else {
                
                    resultLine1 += token1;
                }
                resultLine1 += " ";
            }
            resultLine1.pop_back();
            outData2 << resultLine1 << endl;

            outData3 << "v " << i << " " << int(line[0]);
            if(line.size() > 1)
            {
                outData3 << int(line[1]);
            }
            outData3 << endl;
        }
        getline(inData, line);
        stringstream ssNew(line);
        ssNew >> edges;
        for(int i = 0; i < edges; i++)
        {
            getline(inData, line);
            outData1 << "u " << line << endl;
            outData2 << "e ";
            istringstream iss2(line);
            string token2, resultLine2;
            while (iss2 >> token2)
            {
                if (iss2.eof()) {
                
                    if (symbolsToInt.find(token2) == symbolsToInt.end()) {
                        
                        symbolsToInt[token2] = nextInt++;
                    }

                    resultLine2 += to_string(symbolsToInt[token2]);
                } else {
                
                    resultLine2 += token2;
                }
                resultLine2 += " ";
            }
            resultLine2.pop_back();
            outData2 << resultLine2 << endl;
            outData3 << "e " << line << endl;
        }
        getline(inData, line);
        outData2 << endl;
    }
    inData.close();
    outData1.close();
    outData2.close();
    outData3.close();
    outData4.open("noOfGraphs.txt");
    std::cout << "Number of graphs " << lines << endl;
    outData4 << lines << endl;
    outData4.clear();
    return 0;
}