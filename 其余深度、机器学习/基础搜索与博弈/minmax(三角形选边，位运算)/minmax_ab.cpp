#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <unordered_map>

using namespace std;

/**
 * 由于这是个选边博弈，主要以边为单位
 * 此外，由于A先选边，近似考虑其为最大化玩家，B为最小化玩家
 * State维护可选边列表，并对其按潜在得分排序 加速剪枝
 * State维护完全占有三角形列表，优化 _is_triangle_been_fully_occupied
 */
class TriangleGame {
private:
    long EDGES_BOUND     = 17 + 1;
    long GAME_OVER_MASK  = (1 << EDGES_BOUND) - 1;   // 游戏结束位掩码
    vector<long> vertexs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<pair<long, long>> valid_edges = {
        {1, 2}, {1, 3}, {2,  3}, {2, 4}, {2, 5}, {3,  5},
        {3, 6}, {4, 5}, {5,  6}, {4, 7}, {4, 8}, {5,  8},
        {5, 9}, {6, 9}, {6, 10}, {7, 8}, {8, 9}, {9, 10}
    };
    vector<vector<long>> triangles = {  // 存的是valid_edges的索引
        { 0,  1,  2}, // 123
        { 3,  4,  7}, // 245
        { 2,  4,  5}, // 235
        { 5,  6,  8}, // 356
        { 9, 10, 15}, // 478
        { 7, 10, 11}, // 458
        {11, 12, 16}, // 589
        { 8, 12, 13}, // 569
        {13, 14, 17}  // 69 10
    };
    vector<vector<long>> edgeMap;
    vector<long> triangle_masks;
    unordered_map<long, long> score_cache; // 缓存边的潜在得分
    

    struct State {
        int edges;       // 已选边的位掩码 （第k位为1表示第k条边已选）
        int triangle;    // 已触发三角形的位掩码
        int max_score;   // A的得分
        int min_score;   // B的得分
        bool max_turn;   // 是否轮到A
        State(long e, long t, long max_score, long min_score, bool turn)
            : edges(e), triangle(t), max_score(max_score), min_score(min_score), max_turn(turn) {}
    };

    void __init_edgeMap() {
        edgeMap.resize(EDGES_BOUND);
        triangle_masks.resize(triangles.size());
        for (size_t ti = 0; ti < triangles.size(); ti++) {
            long mask = 0;
            for (long e : triangles[ti]) {
                edgeMap[e].push_back(ti);
                mask |= (1 << e);
            }
            triangle_masks[ti] = mask;
        }
    }

    void __initialize_state(long& m, long& edges, long& triangle, long& max_score, long& min_score, bool& max_turn) {
        cin >> m;
        for (long i = 0; i < m; i++) {
            long u, v;
            cin >> u >> v;
            pair<long, long> edge = {min(u, v), max(u, v)};
            auto it = find(valid_edges.begin(), valid_edges.end(), edge);
            if (it == valid_edges.end()) {
                cerr << "Invalid edge input: (" << u << ", " << v << ")" << endl;
                return;
            }
            long e_idx = distance(valid_edges.begin(), it);

            edges      = __choose(edges, e_idx);
            long score = __calc_score(e_idx, State(edges, triangle, max_score, min_score, max_turn));

            if (max_turn) max_score += score;
            else          min_score += score;

            triangle = __update_triangles(e_idx, edges, triangle);

            // 如果没有新的三角形被占有，轮到对方
            if (score == 0) {
                max_turn = !max_turn;
            }
        }
    }
    /**
     * 得分当且仅当边 e 选择后占有了新的三角形，且其之前未被占有
     */
    long __calc_score(long e, const State& s) {
        long score = 0;
        for (long ti : edgeMap[e]) {
            if (__has_not_been_choosen(s.triangle, ti)) {
                // 选中e后的边集合
                long edges_after = s.edges | (1 << e);
                if ((edges_after & triangle_masks[ti]) == triangle_masks[ti]) {
                    score++;
                }
            }
        }
        return score;
    }

    bool __is_game_over(long edges) {
        return edges == GAME_OVER_MASK;
    }

    bool __has_not_been_choosen(long mask, long index) {
        return !(mask & (1 << index));
    }

    bool __has_been_choosen(long mask, long index) {
        return mask & (1 << index);
    }

    long __choose(long mask, long index) {
        return mask | (1 << index);
    }

    bool __is_triangle_been_fully_occupied(long triangle, long edges) {
        return (edges & triangle_masks[triangle]) == triangle_masks[triangle];
    }

    /**
     * 除了当前边，其它的必须存在
     */
    bool __is_triangle_been_partly_occupied(long triangle, long edges, long current_edge) {
        long edge1 = triangles[triangle][0];
        long edge2 = triangles[triangle][1];
        long edge3 = triangles[triangle][2];

        return (edge1 == current_edge || __has_been_choosen(edges, edge1)) &&
               (edge2 == current_edge || __has_been_choosen(edges, edge2)) &&
               (edge3 == current_edge || __has_been_choosen(edges, edge3));
    }

    long __update_triangles(long e_idx, long edges, long cur_triangle_state) {
        for (long triangle : edgeMap[e_idx]) {
            if (__is_triangle_been_fully_occupied(triangle, edges) && __has_not_been_choosen(cur_triangle_state, triangle)) {
                cur_triangle_state = __choose(cur_triangle_state, triangle);
            }
        }
        return cur_triangle_state;
    }

    /**
     * α：当前最大化玩家可以确保的最小值
     * β：当前最小化玩家可以确保的最大值
     */
    bool __should_trim(long& alpha, long& beta, long& value, long child_value, bool max_turn) {
        if (max_turn) {
            value = max(value, child_value);
            alpha = max(alpha, value);
            return value >= beta;       // β剪枝
        } else {
            value = min(value, child_value);
            beta  = min(beta, value);
            return value <= alpha;      // α剪枝
        }
    }

    static bool compareEdges(const pair<long, long>& a, const pair<long, long>& b) {
        return a.second > b.second;
    }

    long min_max(State state, long alpha, long beta, bool max_turn) {
        // 边被全部选完，博弈终止
        if (__is_game_over(state.edges)) {
            return state.max_score - state.min_score;
        }

        //找到所有可用边
        vector<pair<long, long>> available_edges;
        for (long i = 0; i < EDGES_BOUND; i++) {
            if (__has_not_been_choosen(state.edges, i)) {
                long h = __calc_score(i, state);
                available_edges.emplace_back(i, h);
            }
        }

        // 按潜在得分预排序（加速剪枝）
        sort(available_edges.begin(), available_edges.end(), compareEdges);

        long value = max_turn ? LONG_MIN : LONG_MAX;

        for (const auto& edgePair : available_edges) {
            long edge = edgePair.first;
            long choose_edge        = __choose(state.edges, edge);
            long cur_triangle_state = state.triangle;
            long max_score = state.max_score;
            long min_score = state.min_score;

            long score = __calc_score(edge, state);
            if (score > 0) {        // 有得分（可以占有新的三角形）
                if (max_turn) max_score += score;  // 更新A的得分
                else          min_score += score;  // 更新B的得分
                cur_triangle_state = __update_triangles(edge, choose_edge, cur_triangle_state);
            }

            // 确定下一回合是否轮到当前玩家（如果形成一个三角形，则三角形归他所有，而且还必须再走一步）
            bool next_turn = (score > 0) ? max_turn : !max_turn;
            State cur_state(choose_edge, cur_triangle_state, max_score, min_score, next_turn);
            long child_value = min_max(cur_state, alpha, beta, next_turn);

            // 更新值并进行α-β剪枝
            if (__should_trim(alpha, beta, value, child_value, max_turn)) break;
        }

        return value;
    }

public:
    TriangleGame() : edgeMap(EDGES_BOUND) {
        __init_edgeMap();
    }

    void startUp() {
        long m;
        long edges     = 0;
        long triangle  = 0;
        long max_score = 0;
        long min_score = 0;
        bool max_turn  = true;

        __initialize_state(m, edges, triangle, max_score, min_score, max_turn);

        State initial(edges, triangle, max_score, min_score, max_turn);
        long result = min_max(initial, LONG_MIN, LONG_MAX, initial.max_turn);

        cout << (result > 0 ? "A wins." : "B wins.") << endl;
    }
};

int main() {
    TriangleGame game;
    game.startUp();
    return 0;
}