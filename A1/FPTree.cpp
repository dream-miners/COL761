// code getting long item sets

#include<iostream>
#include<vector>
#include<list>
#include <fstream>
#include <cstdlib> 
#include <string>
#include<algorithm>
using namespace std;

ofstream outData;

struct node
{
    int name;                     // name of the item
    int freq;                // frequency of the item
    int depth;                    // depth of the node in the FP Tree
    int index;                    // index of chuldren vector of parent
    node *parent;               // parent node
    vector<node *> children;    // children nodes
    node *next;

    node()
    {
        this->next = nullptr;
    }
}*root;

class FPTree
{
    public:
        node* root;
        int numTrans;
        vector<node *> path;
        vector<node *> head;        // vector of pointers to the first item of linked list in the FP tree
        vector<node *> currentNode; // vector of pointers to the last item of linked list in the FP tree

        FPTree()
        {
            root = new node();
            root->name = 0;
            root->freq = 0;
            root->depth = 0;
            root->index = 0;
            root->parent = nullptr;
            root->children.push_back(nullptr);
            numTrans = 0;
        }

        ~FPTree()
        {
            deleteTree(root);
        }

        void deleteTree(node *node)
        {
            if(node==nullptr)
                return;
            else
            {
                for(int i=0; i<node->children.size(); i++)
                {
                    deleteTree(node->children[i]);
                }
            }
            delete node;
        }

        void makeFPnode(int name, node *parent, int i, int freq = 1)
        {
            node *child = new node;         // new FP node
            child->name = name;                 // assigning name
            child->freq = freq;                    // assigning frequency
            child->depth = parent->depth + 1;   // assigning depth
            child->index = i;
            child->parent = parent;
            child->children.push_back(nullptr); // initialising children nodes
            child->next = nullptr;

            parent->children.pop_back();            // removing last element which is null pointer
            parent->children.push_back(child);      // pushing new node at the last position
            parent->children.push_back(nullptr);    // pushing null pointer at the end
            child = nullptr;
            delete child;
        }

        void makeFPBranch(vector<int> items, node *rootNode, int freq=1)
        {
            numTrans++;
            node *current = new node;           // creating a new node
            current = rootNode;                 // current node points to root

            for(int i=0; i<items.size(); i++)
            {
                int j;
                for(j = 0; current->children[j]!=nullptr; j++)
                {
                    if(items[i] == current->children[j]->name)
                    {
                        current->children[j]->freq = current->children[j]->freq + freq;
                        // cout << current->children[j]->name << endl;
                        break;
                    }
                }
                if(current->children[j]==nullptr)
                {
                    makeFPnode(items[i], current, j, freq);
                    int k;
                    for(k=0; k<currentNode.size(); k++)
                    {
                        if(current->children[j]->name==currentNode[k]->name)
                        {
                            currentNode[k]->next = current->children[j];
                            currentNode[k] = currentNode[k]->next;
                            break;
                        }
                    }
                    if(k==currentNode.size())
                    {
                        head.push_back(new node());
                        currentNode.push_back(current->children[j]);
                        head[k]->next = currentNode[k];
                    }
                    // cout << current->children[j]->name << " ";
                }
                // cout << current->children[j]->name << endl;
                current = current->children[j];
            }
            // cout << endl;
            current = nullptr;
            delete current;
            // free(current);
        }

        void printFPTree(node *fpnode)
        {
            // path printing function
            node *current = fpnode;
            if(!isNull(current))
            {
                if(current->name!=-1)
                    path.push_back(current);
                // cout << current->name << " ";
                while(!isNull(current->children[0]))
                {
                    current = current->children[0];
                    path.push_back(current);
                    // cout << current->name << endl;
                }

                // Leaf node
                for(int i=0; i<path.size(); i++)
                    cout << path[i]->name << ": " << path[i]->freq << " ";
                cout << endl;

                // check if sibling is present
                int index = current->index;
                if(!isNull(current->parent))
                {
                    current = current->parent;
                    path.pop_back();
                    while (isNull(current->children[index+1]))
                    {
                        if(!isNull(current->parent))
                        {
                            index = current->index;
                            current = current->parent;
                            path.pop_back();
                        }
                        else
                            return;
                    }
                    printFPTree(current->children[index+1]);
                }
            }
            else
                return;
        }

        bool isNull(node* node)
        {
            // check if fpnode is a null pointer
            bool ans = (node==nullptr)? true : false;
            return ans;
        }
};

void merge(vector<node *>& arr, unsigned long int left, unsigned long int middle, unsigned long int right)
{
    unsigned long int n1 = middle - left + 1;
    unsigned long int n2 = right - middle;

    vector<node *> leftArray(n1);
    vector<node *> rightArray(n2);

    for (int i = 0; i < n1; i++)
        leftArray[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        rightArray[j]= arr[middle + 1 + j];

    int i = 0;
    int j = 0;
    unsigned long int k = left;

    while (i < n1 && j < n2) {
        if (leftArray[i]->name >= rightArray[j]->name)
        {
            arr[k] = leftArray[i];
            i++;
        }
        else
        {
            arr[k] = rightArray[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = leftArray[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = rightArray[j];
        j++;
        k++;
    }
}

void mergeSort(vector<node *>& arr, unsigned long int left, unsigned long int right)
{
    if (left < right) {
        unsigned long int middle = left + (right - left) / 2;

        mergeSort(arr, left, middle);
        mergeSort(arr, middle + 1, right);

        merge(arr, left, middle, right);
        // for(int i=0; i<right; i++)
        // {
        //     cout << arr[i]->name << endl;
        // }
    }
}

vector<int > makePath(node* leaf)
{
    vector<int> pathNow;
    // pathNow.push_back(leaf->name);
    while(leaf->parent->name!=0)
    {
        leaf = leaf->parent;
        pathNow.push_back(leaf->name);
        // cout << leaf->name << " ";
    }
    // cout << endl;
    reverse(pathNow.begin(), pathNow.end());
    return pathNow;
}

FPTree makeCondTree(node* start)
{
    FPTree condTree;
    node* here = start;
    int freq = here->freq;
    for(int i=0; here!=nullptr; i++)
    {
        vector<int> newPath = makePath(here);
        condTree.makeFPBranch(newPath, condTree.root, freq);
        here = here->next;
    }
    return condTree;
}

int getTotalFreq(node *start)
{
    int totalFreq = 0;
    while(start->next!=nullptr)
    {
        totalFreq = totalFreq + start->freq;
        start = start->next;
    }
    totalFreq = totalFreq + start->freq;
    return totalFreq;
}

void mineFrequentItems(node* currentNode, int minSupport, vector<int>& frequentItems)
{
    if (!currentNode)
    {
        return;
    }

    int totalFrequency = getTotalFreq(currentNode);
    if (totalFrequency >= minSupport && currentNode->name != 0)
    {
        int i;
        for(i=0; i<frequentItems.size(); i++)
        {
            if(currentNode->name == frequentItems[i])
                break;
        }
        if(i==frequentItems.size())
            frequentItems.push_back(currentNode->name);
    }

    for(int i=0; i<currentNode->children.size(); i++)
    {
        mineFrequentItems(currentNode->children[i], minSupport, frequentItems);
    }
}

void recursiveMining(node *start, int support, vector<int> &freqItems)
{
    if(start->name!=0)
    {   
        // cout << start->name << endl;
        FPTree condTree = makeCondTree(start);
        vector<int> currentSet;
        mineFrequentItems(condTree.root, support, currentSet);
        freqItems.push_back(start->name);
        if(currentSet.size()==0)
            return;
        unsigned long int n = condTree.head.size();
        // for (int i=0; i < freqItems.size(); i++)
        // {
        //     outData << freqItems[i] << " "<< start->name << endl;
        // }
        // reverse(freqItems.begin(), freqItems.end());
        // cout << "Frequent items in the conditional FP Tree of item " << start->name << ":\n";
        for(int i=0; i<n; i++)
        {
            condTree.head[i] = condTree.head[i]->next;
        }
        mergeSort(condTree.head, 0, n-1);

        for(int i=0; i<n; i++)
        {
            if(getTotalFreq(condTree.head[i])>=support)
            {
                // cout << condTree.head[i]->name << endl;
                recursiveMining(condTree.head[i], support, freqItems);
                break;
            }
        }
        condTree.deleteTree(condTree.root);
    }
    return;
}


int main()
{
    FPTree fptree;
    fstream inData;
    int num, oldNum = 0;

    vector<int> items;
    inData.open("D_small.dat");
    if(!inData)
    {
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    inData >> num;

    while (!inData.eof())
    {
        while(num > oldNum)
        {
            oldNum = num;
            items.push_back(num);
            inData >> num;
        }
        fptree.makeFPBranch(items, fptree.root);
        items.clear();
        oldNum = 0;
    }
    inData.close();
    outData.open("freqSets.txt");

    // vector<int> item1, item2, item3, item4;
    // for(int i=0; i<5; i++)
    // {
    //     item1.push_back(i+1);
    //     item2.push_back(2*i+1);
    //     item3.push_back(i+2);
    //     item4.push_back(2*i+2);
    // }

    // fptree.makeFPBranch(item4, fptree.root);
    // fptree.makeFPBranch(item1, fptree.root);
    // fptree.makeFPBranch(item2, fptree.root);
    // fptree.makeFPBranch(item3, fptree.root);
    // cout << "Original FP Tree\n";
    // fptree.printFPTree(fptree.root);

    unsigned long int n = fptree.head.size();

    for(int i=0; i<n; i++)
    {
        fptree.head[i] = fptree.head[i]->next;
        // cout << fptree.head[i]->name << " ";
    }
    // cout << endl;

    mergeSort(fptree.head, 0, n-1);

    int support = 0.5*fptree.numTrans;

    int i = 5;
    vector<int> freq;
    outData << fptree.head[0]->name+1 << endl;
    for(int i=0; i < n; i++)
    {
        if(getTotalFreq(fptree.head[i]) >= support)
        {
            recursiveMining(fptree.head[i], support, freq);
            for(int i=0; i<freq.size(); i++)
            {
                // cout << freq[i] << " ";
                outData << freq[i] << " ";
            }
            // cout << endl;
            outData << endl;
            freq.clear();
        }
    }
    outData.close();

    return 0;
}
