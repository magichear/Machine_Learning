#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <dirent.h> // 用于手动遍历文件
#include <chrono>

using namespace std;

struct Grid {
    vector<vector<int>> state;  // 当前状态
    pair<int, int> zero;        // 空格的位置
    int g, h, f;                // g(n): 已走步数, h(n): 估计剩余步数, f(n): 总估计代价
    Grid* pre;                  // 前驱节点，用于记录路径

    // 初始化状态和前驱节点
    Grid(const vector<vector<int>>& state, Grid* pre = nullptr) 
        : state(state), pre(pre) {
        // 找到空格的位置
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (state[i][j] == 0) {
                    zero = {i, j};
                    break;
                }
            }
        }
        g = pre ? pre->g + 1 : 0; // 如果有前驱节点，g值加1
        h = 0;                    // h值将在update中计算
        f = g + h;                // 初始化f值
    }

    // 更新h值（曼哈顿距离 + 线性冲突）并重新计算f值
    void update(const vector<vector<int>>& target) {
        h = 0;
        int linear_conflict = 0;
    
        // 计算曼哈顿距离
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                int value = state[i][j];
                if (value != 0) {
                    // 找到目标位置
                    for (int ti = 0; ti < 3; ++ti) {
                        for (int tj = 0; tj < 3; ++tj) {
                            if (target[ti][tj] == value) {
                                h += abs(ti - i) + abs(tj - j);
    
                                // 检查线性冲突
                                if (ti == i) { // 同一行
                                    for (int k = j + 1; k < 3; ++k) {
                                        int next_value = state[i][k];
                                        if (next_value != 0) {
                                            for (int tti = 0; tti < 3; ++tti) {
                                                for (int ttj = 0; ttj < 3; ++ttj) {
                                                    if (target[tti][ttj] == next_value && tti == i && ttj < tj) {
                                                        linear_conflict += 2; // 每个冲突增加2步
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if (tj == j) { // 同一列
                                    for (int k = i + 1; k < 3; ++k) {
                                        int next_value = state[k][j];
                                        if (next_value != 0) {
                                            for (int tti = 0; tti < 3; ++tti) {
                                                for (int ttj = 0; ttj < 3; ++ttj) {
                                                    if (target[tti][ttj] == next_value && ttj == j && tti < ti) {
                                                        linear_conflict += 2; // 每个冲突增加2步
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    
        h += linear_conflict; // 将线性冲突代价加到h值中
        f = g + h; // 更新f值
    }

    // 扩展当前节点，生成所有可能的下一步状态
    vector<Grid> expand(const vector<vector<int>>& target) {
        vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; // 上下左右移动
        vector<Grid> neighbors;

        for (auto [di, dj] : directions) {
            int ni = zero.first + di, nj = zero.second + dj;
            if (ni >= 0 && ni < 3 && nj >= 0 && nj < 3) { // 检查边界
                vector<vector<int>> new_state = state;
                swap(new_state[zero.first][zero.second], new_state[ni][nj]); // 交换空格与目标位置
                Grid neighbor(new_state, this); // 创建新节点
                neighbor.update(target);        // 更新新节点的h值和f值
                neighbors.push_back(neighbor);  // 加入邻居列表
            }
        }
        return neighbors;
    }

    // 用于优先队列（f值小的优先）
    bool operator<(const Grid& other) const {
        return f > other.f; // 优先队列是大顶堆，因此这里反向比较
    }
};

class AStarSolver {
private:
    vector<vector<int>> initial_state; // 初始状态
    vector<vector<int>> goal_state;    // 目标状态

    // 将二维矩阵展平为一维数组
    vector<int> flatten(const vector<vector<int>>& matrix) {
        vector<int> flat;
        for (const auto& row : matrix) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
        return flat;
    }

    // 计算逆序数
    int calc_inverse_nums(const vector<int>& numbers) {
        vector<int> arr;
        for (int num : numbers) {
            if (num != 0) arr.push_back(num); // 忽略空格
        }
        return __merge_sort_and_count(arr, 0, arr.size() - 1);
    }

    // 借助归并排序计算逆序数
    int __merge_sort_and_count(vector<int>& arr, int left, int right) {
        if (left >= right) return 0;

        int mid = left + (right - left) / 2;
        int count = __merge_sort_and_count(arr, left, mid) +
                    __merge_sort_and_count(arr, mid + 1, right);

        vector<int> temp;
        int i = left, j = mid + 1;

        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp.push_back(arr[i++]);
            } else {
                temp.push_back(arr[j++]);
                count += mid - i + 1; // 统计逆序数
            }
        }

        while (i <= mid) temp.push_back(arr[i++]);
        while (j <= right) temp.push_back(arr[j++]);

        for (int k = left; k <= right; ++k) {
            arr[k] = temp[k - left];
        }

        return count;
    }

    // 将状态矩阵转换为字符串，用于哈希表存储
    string state_to_string(const vector<vector<int>>& state) {
        stringstream ss;
        for (const auto& row : state) {
            for (int num : row) {
                ss << num << ",";
            }
        }
        return ss.str();
    }

public:
    AStarSolver(const vector<vector<int>>& initial_state, const vector<vector<int>>& goal_state)
        : initial_state(initial_state), goal_state(goal_state) {}

    // 判断问题是否有解
    bool is_solvable() {
        return (calc_inverse_nums(flatten(initial_state)) & 1) == 
               (calc_inverse_nums(flatten(goal_state)) & 1);
    }

    // A*
    int solve() {
        if (!is_solvable()) return -1; // 无解

        priority_queue<Grid> open_list;          // 优先队列，存储待处理节点
        unordered_map<string, bool> closed_dict; // 哈希表，存储已访问状态

        Grid start_node(initial_state); // 创建起始节点
        start_node.update(goal_state);  // 更新起始节点的h值和f值
        open_list.push(start_node);     // 将起始节点加入优先队列

        while (!open_list.empty()) {
            Grid cur_node = open_list.top(); // 取出f值最小的节点
            open_list.pop();

            if (cur_node.h == 0) return cur_node.g; // 如果h值为0，说明已到达目标状态

            string cur_state_key = state_to_string(cur_node.state);
            closed_dict[cur_state_key] = true; // 标记当前状态为已访问

            for (auto& next_node : cur_node.expand(goal_state)) { // 扩展当前节点
                string next_state_key = state_to_string(next_node.state);
                if (closed_dict.find(next_state_key) != closed_dict.end()) continue; // 跳过已访问状态

                open_list.push(next_node); // 将新节点加入优先队列（这里会顺便比较f值）
            }
        }
        return -1; // 未找到解
    }
};

int main() {
    vector<vector<int>> goal_state(3, vector<int>(3));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cin >> goal_state[i][j];
        }
    }

    // 初始状态
    vector<vector<int>> initial_state = {{1, 2, 3}, {4, 5, 6}, {7, 8, 0}};

    // 使用 A* 算法求解
    AStarSolver solver(initial_state, goal_state);
    int result = solver.solve();

    // 输出结果
    if (result == -1) {
        cout << "Impossible" << endl;
    } else {
        cout << result << endl;
    }
    return 0;
}

vector<vector<int>> read_puzzle_file(const string& filepath) {
    ifstream file(filepath);
    vector<vector<int>> puzzle;
    if (file.is_open()) {
        string line;
        getline(file, line); // 读取第一行，表示矩阵大小
        int n = stoi(line);
        for (int i = 0; i < n; ++i) {
            getline(file, line);
            stringstream ss(line);
            vector<int> row;
            int num;
            while (ss >> num) {
                row.push_back(num);
            }
            puzzle.push_back(row);
        }
        file.close();
    } else {
        cerr << "无法打开文件: " << filepath << endl;
    }
    return puzzle;
}

vector<string> get_files_with_prefix(const string& folder_path, const string& prefix) {
    vector<string> filelist;
    DIR* dir = opendir(folder_path.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            string filename = entry->d_name;
            if (filename.find(prefix) == 0) { // 检查前缀
                filelist.push_back(folder_path + "\\" + filename);
            }
        }
        closedir(dir);
    } else {
        cerr << "无法打开目录: " << folder_path << endl;
    }
    return filelist;
}

void test() {
    string folder_path = "D:\\Study_Work\\Electronic_data\\CS\\AAAUniversity\\rgznjc\\Lab\\L1\\8Code\\8puzzle\\8puzzle";
    string file_prefix = "puzzle3x3";

    vector<string> filelist = get_files_with_prefix(folder_path, file_prefix);

    // 开始计时
    auto start_time = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < filelist.size(); ++i) {
        cout << i << " " << filelist[i] << " ";
        vector<vector<int>> initial_state = read_puzzle_file(filelist[i]);

        // 目标状态
        vector<vector<int>> goal_state = {{1, 2, 3}, {4, 5, 6}, {7, 8, 0}};

        // 使用 A* 算法求解
        AStarSolver solver(initial_state, goal_state);
        int result = solver.solve();

        cout << "Moves: " << result << endl;
    }

    // 结束计时
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;

    // 输出总耗时
    cout << "Total time: " << elapsed_time.count() << " seconds" << endl;

    return;
}

/*
int main() {
    test();
    return 0;
}
*/