#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#ifndef FP_TREE_H
#define FP_TREE_H
#include<cmath>

#define pb push_back
#define mp make_pair
#define x first
#define y second

using namespace std;

typedef vector<int> vi;
typedef vector<vi> vvi;
typedef pair<int,int> pii;
typedef vector<pii> vpii;
#endif
#include <memory>
#include<vector>
#include<unordered_map>

using namespace std;
#ifndef UNTITLED_STRUC_DEF_H
#define UNTITLED_STRUC_DEF_H

#endif //UNTITLED_STRUC_DEF_H

#include <map>
#include <vector>
#include <algorithm>

// maping of frequent item sets to integer
map<vector<int>, int> itemMapping;

// sorting vector in ascending order
void merge(vector<int>& arr, int left, int middle, int right) {
    int n1 = middle - left + 1;
    int n2 = right - middle;

    vector<int> leftArray(n1);
    vector<int> rightArray(n2);

    for (int i = 0; i < n1; i++)
        leftArray[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        rightArray[j]= arr[middle + 1 + j];

    int i = 0;
    int j = 0;
    int k = left;

    while (i < n1 && j < n2) {
        if (leftArray[i] <= rightArray[j])
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

// splitting, sorting and then merging a vector
void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int middle = left + (right - left) / 2;

        mergeSort(arr, left, middle);
        mergeSort(arr, middle + 1, right);

        merge(arr, left, middle, right);
    }
}

// modifying vector by removing frequent items and adding new mapped element
vector<int> modifyVector(vector<int> &transaction, vector<int>items, int item)
{
    vector<int> modified;
    int i = 0;
    for(auto item1 : transaction)
    {
        for(auto itr : items)
        {
            if(item1==itr)
            {
                i++;
                break;
            }
        }
        if(i==0)
            modified.push_back(item1);
        i = 0;
    }
    modified.push_back(item);
    return modified;
}


// compressing transaction
vector<int> compress(vector<int> &transaction)
{
    map<vector<int>,int>::iterator itr;
    vector<int> items;
    vector<int> compressed_transaction;
    for(itr=itemMapping.end(); itr!=itemMapping.begin(); --itr)
    {
        if(itr->first.size()>transaction.size())
            continue;
        for(auto item : itr->first)
        {
            if (find(transaction.begin(), transaction.end(), item) == transaction.end())
            {
                break;
            }
            items.push_back(*find(transaction.begin(), transaction.end(), item));
        }

        if(items.size()==itr->first.size())
        {
            compressed_transaction = modifyVector(transaction, items, itr->second);
        }
        items.clear();
    }
    return compressed_transaction;
}

class Bucket{
public:
    int numb;
    int numbC;
    shared_ptr<Bucket> next_same_numb_ptr; // pointer to next same item
    
    unordered_map<int,shared_ptr<Bucket>> child_bucket_dict; // child node pointers
    shared_ptr<Bucket> parent_bucket_ptr;   // pointer to parent node

    Bucket(int numb, shared_ptr<Bucket> parent_bucket_ptr, int numbC=1) {
        this->numb = numb;
        this->numbC = numbC
                ;
        this->parent_bucket_ptr = parent_bucket_ptr;
        this->next_same_numb_ptr=NULL;
    }
};


typedef std::pair<int,int> pair_;

class Tree{
    public:
        unordered_map<int,shared_ptr<Bucket>> dictionary_first_bucket;
        shared_ptr<Bucket> tree_root;
        vector<pair_> vec_freq_num_sorted_dec;
};


double CUT_OFF_PERCENT=2;   // support in percentage
int TOTAL_TRANSACTIONS=0;   // total number of transactions
int REQUIRED_SUPPORT_COUNT=0;   // support in number
int TOTAL_ITEMS=0;  // total items in file
int TOTAL_NODES=0;  // total nodes in FP Tree

unordered_map<int,int> item_idx;
vector<int> idx_item;

vvi frequent_patterns;  //final Answer

vi path_count; // Keeps count of Number of paths through this item
vi node_to_item;
vector<unordered_map<int,int> > track_children_items;  // Keeps track of all children of node - unordered_maps item to Node Number
vector<int> parent_node;
vvi header_table;   // header table for FP tree

int idx=0;
vi prefix_frequent_itemset;

void calculate_support()
{
	double count= CUT_OFF_PERCENT*TOTAL_TRANSACTIONS/100;
	REQUIRED_SUPPORT_COUNT=ceil(count); // support in numbers
}

/*
	Used for sorting <item,count> pair in decreasing order of count
*/
bool COMPARATOR(pii a,pii b)
{
	if(a.y!=b.y)
		return a.y>b.y;
	return a.x<b.x;
}

/*
	Used for sorting itemsets
*/
bool COMPARATOR2(int a,int b)
{
	return a<b;
}

// building FP Tree
void growFP_Tree(vi &itemset)
{
	int cur_node=0;

	for(auto itr:itemset)
	{
		int next_node=-1;

		if(track_children_items[cur_node].find(itr)==track_children_items[cur_node].end())
		{
			track_children_items[cur_node][itr]=TOTAL_NODES;    // Create a new child of Parent
			parent_node.pb(cur_node);   // Assign parent to child
			header_table[itr].pb(TOTAL_NODES);  // Update header Pointers
			node_to_item.pb(itr);   // Reverse Mapping
			next_node=TOTAL_NODES;
			TOTAL_NODES++;
			track_children_items.resize(TOTAL_NODES);
			path_count.pb(0);
		}
		else
			next_node=track_children_items[cur_node][itr];
		cur_node=next_node;
		path_count[cur_node]++;
	}
}

vi local_idx_item;

void constructFP_Tree(string filename)
{
	string line;
	vpii count_Transactions;
	unordered_map<int,int> local_item_idx;
	int temp_count=-1;

/*
	First Pass over dataset
	Setup Frequency table for items
	Required to order the items in itemset
*/

	ifstream read_input;
	read_input.open(filename);
    string tokens;

	while(getline(read_input,line))
	{
		TOTAL_TRANSACTIONS++;

		stringstream check1(line);
	    while(getline(check1, tokens, ' '))
	    {
	    	if(tokens.length()==0)
	    		continue;
	    	int item_num=atoi(tokens.c_str());

	    	if(local_item_idx.find(item_num)==local_item_idx.end())
	    	{
	    		temp_count++;
	    		local_item_idx[item_num]=temp_count;
	    		local_idx_item.pb(item_num);
	    		count_Transactions.pb({temp_count,1});
	    	}
	    	else
	    	{
	    		count_Transactions[local_item_idx[item_num]].y++;
	    	}
	    }
    }
	read_input.close();
	calculate_support();
	#ifdef DEBUG
		//cout<<"First Pass Over"<<endl;
 	#endif

/*
	Filtering Items below Support Threshold
	Giving them idxes based on their count
	0-> Highest Count
*/
	sort(count_Transactions.begin(),count_Transactions.end(),COMPARATOR);
	for(auto itr:count_Transactions)
	{
		if(itr.y<REQUIRED_SUPPORT_COUNT)
			break;
		idx_item.pb(local_idx_item[itr.x]);
		item_idx[local_idx_item[itr.x]]=TOTAL_ITEMS;
		TOTAL_ITEMS++;
	}
	#ifdef DEBUG
		//cout<<"Filtering Over"<<endl;
 	#endif

/*
	Setup Data Structures for FP_Tree
	Initialise Header Table
	Make a Root Node
*/	
	header_table.resize(TOTAL_ITEMS);
	parent_node.pb(-1);
	track_children_items.resize(1);
	node_to_item.pb(-1);
	path_count.pb(0);
	TOTAL_NODES++;
/*
	Perform second pass over the data
	Grow FP_Tree simultaneously
*/
	read_input.open(filename);
	vi itemset;

	while(getline(read_input,line))
	{
		stringstream check1(line);
		itemset.resize(0);

	    while(getline(check1, tokens, ' '))
	    {
	    	if(tokens.length()==0)
	    		continue;
			int item_num=atoi(tokens.c_str());

	    	if(item_idx.find(item_num)==item_idx.end())
	    		continue;
	    	itemset.pb(item_idx[item_num]);
	    }
	    if(itemset.size()==0)
	    	continue;
	    sort(itemset.begin(),itemset.end(),COMPARATOR2);

    #ifdef DEBUG
		//cout<<"Begin grow"<<endl;
	#endif
	    growFP_Tree(itemset);
    #ifdef DEBUG
		//cout<<"End Grow"<<endl;
	#endif
    }

	read_input.close();
	#ifdef DEBUG
		//cout<<"Second pass over"<<endl;
 	#endif
}

void augmentFP_set()
{
	vi frequent_item;
	int x=frequent_patterns.size();
	frequent_patterns.pb(frequent_item);
	frequent_patterns[x].resize(idx);
	for(int i=0;i<idx;i++)
		frequent_patterns[x][i]=prefix_frequent_itemset[i];	
}

unordered_map<int,int> global_counter; //   unordered_maps modified item number to their count in conditional tree
unordered_map<int,int> new_header_helper;   // stores filtered nodes
unordered_map<int,int> reset_helper;    // an auxillary unordered_map - can be removed potentially

void reset_path(int node,vpii &storeFP_counts)
{
	int temp=path_count[node];
	node=parent_node[node];

	while(node)
	{
		if(reset_helper.find(node)==reset_helper.end())
		{
			reset_helper[node]=1;
			storeFP_counts.pb({node,path_count[node]});
			path_count[node]=temp;
		}
		else
		{
			path_count[node]+=temp;
		}
		global_counter[node_to_item[node]]+=temp;
		node=parent_node[node];
	}
}

void createHeaderTable(int node,vvi &newHeaderTable,vpii &storeFP_parents)
{
	node=parent_node[node];
	while(node)
	{
		if(new_header_helper.find(node_to_item[node])!=new_header_helper.end())
		{
			int x=new_header_helper[node_to_item[node]];
			if(reset_helper.find(node)==reset_helper.end())
			{
				reset_helper[node]=1;
				newHeaderTable[x].pb(node);
				int temp=parent_node[node];
				while(temp && new_header_helper.find(node_to_item[temp])==new_header_helper.end())
				{
					temp=parent_node[temp];
				}
				if(temp!=parent_node[node])
				{
					storeFP_parents.pb({node,parent_node[node]});
					parent_node[node]=temp;
				}
			}
		}
		node=parent_node[node];
	}
}

void dfs_mineFP(vi &conditional_leaves)
{
	augmentFP_set();

	vpii storeFP_counts;
	vpii storeFP_parents;

	global_counter.clear();
	new_header_helper.clear();
	reset_helper.clear();

	for(auto itr:conditional_leaves)
	{
		reset_path(itr,storeFP_counts);
	}

	int newHeaderCount=0;
	vi temp_nodes;	// Stores item values

	for(auto itr:global_counter)
	{
		if(itr.y<REQUIRED_SUPPORT_COUNT)
			continue;
		new_header_helper[itr.x]=newHeaderCount;
		newHeaderCount++;
		temp_nodes.pb(itr.x);
	}
	vvi newHeaderTable;
	newHeaderTable.resize(newHeaderCount);

	reset_helper.clear();
	for(auto itr:conditional_leaves)
	{
		createHeaderTable(itr,newHeaderTable,storeFP_parents);
	}
	int x=newHeaderTable.size();
	for(int i=x-1;i>=0;i--)
	{
		prefix_frequent_itemset[idx]=temp_nodes[i];
		idx++;
		dfs_mineFP(newHeaderTable[i]);
		idx--;
	}

	for(auto itr:storeFP_counts)
	{
		path_count[itr.x]=itr.y;
	}
	for(auto itr:storeFP_parents)
	{
		parent_node[itr.x]=itr.y;
	}
}

map<int, vector<vector<int>>> freq_map;

void mine_FrequentPatterns()
{
	#ifdef DEBUG
		//cout<<"Hey"<<endl;
 	#endif
	prefix_frequent_itemset.resize(TOTAL_ITEMS);

	for(int i=TOTAL_ITEMS-1;i>=0;i--)
	{
	#ifdef DEBUG
		//cout<<i<<endl;
 	#endif
		prefix_frequent_itemset[idx]=i;
		idx++;
		dfs_mineFP(header_table[i]);
		idx--;
	}
}

void print_FrequentPatterns()
{
	vector <vector <int>> outer;

	int freq = path_count[header_table
	[frequent_patterns[0][0]][0]];

	int oldFreq = 0;
    for (auto itr : frequent_patterns)
    {
		oldFreq = freq;
		freq = path_count[header_table[itr[0]][0]];
		if(freq == oldFreq)
		{
			vector<int> inner;
			for(auto itr2 : itr)
			{
				inner.push_back(idx_item[itr2]);
			}
			outer.push_back(inner);
		}
		else
		{
			freq_map[oldFreq] = outer;
			outer.clear();
		}
    }
}

int main()
{
	cout << "Please enter the path including the file name: ";
	string filename;
	cin >> filename;
	constructFP_Tree(filename);
    cout << "FP Tree constructed\n";

#ifdef SIMPLE_DEBUG
	//cout<<"Constructed FP Tree"<<endl;
#endif
	mine_FrequentPatterns();
    cout << "Frequent items mined.\n";
#ifdef SIMPLE_DEBUG
#endif
	print_FrequentPatterns();
	map<int, vector<vector<int>>> frequent_items_map;
    for (auto itr : frequent_patterns)
    {
        int frequency = path_count[header_table[itr[0]][0]];
        frequent_items_map[frequency].push_back(itr);
    }

	// iterator for traversing map for frequent item replacement
    map<int, vector<vector<int>>>::iterator it;
	ofstream outData;

    int lastItem = local_idx_item.size();
    int i=0;

	// finding longest item set 
    for(it = freq_map.end(); it!=freq_map.begin(); --it)
	{
		if(it->first >= REQUIRED_SUPPORT_COUNT)
		{
			int length = it->second[0].size();
			int index = 0;
			
			for(int i=1; i<it->second.size(); i++)
			{
				if(it->second[i].size() > length)
				{
					length = it->second[i].size();
					index = i;
				}
			}

            mergeSort(it->second[index], 0, it->second[index].size()-1);
			itemMapping[it->second[index]] = lastItem + i;
            i++;
		}
	}

    // printing mapping data
    outData.open("mapping.text");

    map<vector<int>, int>::iterator itr;
    for(itr=itemMapping.begin(); itr!=itemMapping.end(); ++itr)
    {
        outData << itr->second << " ";
        for(auto item : itr->first)
        {
            outData << item << " ";
        }
        outData << endl;
    }
    outData.close();
    cout << "\"mapping.txt\" created.\nCompressing data\n";

    ifstream inData;
    string line;
    // file for writing compressed data
    outData.open("compressedData.txt");

    inData.open(filename);
    vector<int> transaction; // Create a new vector for each line
    vector<int> modified_trans; // vector for compressed transaction
    while (getline(inData, line))
    {
        stringstream ss(line);

        int num;
        while (ss >> num)
        {
            transaction.push_back(num);
        }
        modified_trans = compress(transaction);
        if(modified_trans.size()!=0)
        {
            for(auto item : modified_trans)
            {
                outData << item << " ";
            }
            outData << endl;
        }
        else
        {
            for(auto item : transaction)
            {
                outData << item << " ";
            }
            outData << endl;
        }
        // clearing transaction for next line of input
        transaction.clear();
    }
    inData.close();
    outData.close();
    cout << "\"compressedData.txt\" created.\nData compressed\n";

#ifdef SIMPLE_DEBUG
	//cout<<"End"<<endl;
#endif
	return 0;
}