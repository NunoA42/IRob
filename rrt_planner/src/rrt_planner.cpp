
#include <rrt_planner/rrt_planner.h>

namespace rrt_planner {

    RRTPlanner::RRTPlanner(costmap_2d::Costmap2DROS *costmap, 
            const rrt_params& params) : params_(params), collision_dect_(costmap) {

        costmap_ = costmap->getCostmap();
        map_width_  = costmap_->getSizeInMetersX();
        map_height_ = costmap_->getSizeInMetersY();

        random_double_x.setRange(-map_width_, map_width_);
        random_double_y.setRange(-map_height_, map_height_);

        nodes_.reserve(params_.max_num_nodes);
    }

    bool RRTPlanner::planPath() {

        // clear everything before planning
        nodes_.clear();

        // Start Node
        createNewNode(start_, -1);

        double *p_rand, *p_new;
        Node nearest_node;

        for (unsigned int k = 1; k <= params_.max_num_nodes; k++) {

            p_rand = sampleRandomPoint();
            nearest_node = nodes_[getNearestNodeId(p_rand)];
            p_new = extendTree(nearest_node.pos, p_rand); // new point and node candidate

            if (!collision_dect_.obstacleBetween(nearest_node.pos, p_new)) {
                createNewNode(p_new, nearest_node.node_id);

            } else {
                continue;
            }

            if(k > params_.min_num_nodes) {
                
                if(computeDistance(p_new, goal_) <= params_.goal_tolerance){
                    return true;
                }
            }
        }

        return false;
    }

    int RRTPlanner::getNearestNodeId(const double *point) {

        /**************************
         * Implement your code here
         **************************/
        int nearest_id = 0;
        double min_distance = std::numeric_limits<double>::max();
    
        for (size_t i = 0; i < nodes_.size(); i++) {
            double dist = computeDistance(nodes_[i].pos, point);
            if (dist < min_distance) {
                min_distance = dist;
                nearest_id = i;
                
                // Early termination if very close
                if (dist < params_.step * 0.5) {
                break;
            }
            }
        }
    
        return nearest_id;

    }

    void RRTPlanner::createNewNode(const double* pos, int parent_node_id) {

        Node new_node;

        /**************************
         * Implement your code here
         **************************/
        new_node.pos[0] = pos[0];
        new_node.pos[1] = pos[1];
        new_node.node_id = nodes_.size(); // Current size will be this node's ID
        new_node.parent_id = parent_node_id;
        nodes_.emplace_back(new_node);
        
    }

    double* RRTPlanner::sampleRandomPoint() {
        /**************************
         * Implement your code here
         **************************/
        // 3% chance to sample goal directly
        const double goal_bias_prob = 0.030;
        double prob = ((double)rand() / RAND_MAX);

        // Sample in a region around the goal
        // if (prob < goal_bias_prob) {
        //     double radius = 1.0;  // meters
        //     double angle = ((double)rand() / RAND_MAX) * 2 * M_PI;
        //     double r = sqrt((double)rand() / RAND_MAX) * radius;
        //     rand_point_[0] = goal_[0] + r * cos(angle);
        //     rand_point_[1] = goal_[1] + r * sin(angle);
        // }   

        // Sample the goal
        if (prob < goal_bias_prob) {
            rand_point_[0] = goal_[0];
            rand_point_[1] = goal_[1];
        } else {
            rand_point_[0] = random_double_x.generate();
            rand_point_[1] = random_double_y.generate();
        }
        

        return rand_point_;
    }

    double* RRTPlanner::extendTree(const double* point_nearest, const double* point_rand) {
        // Adaptive step based on distance to goal
        double dist_to_goal = computeDistance(point_nearest, goal_);
        double adaptive_step;
        
    // if (dist_to_goal < 0.5) {
    //     adaptive_step = params_.step * 0.65; //0.25;  // Very small near goal
    // } else if (dist_to_goal < 2.0) {
    //     adaptive_step = params_.step; //*0.75;
    // } else {
    //     adaptive_step = params_.step * 2; //1.5;  // Larger steps far away
    // }
        if (dist_to_goal < 2) {
            adaptive_step = params_.step;
        }else {
            adaptive_step = params_.step * 2; //1.5;  // Larger steps far away
        }

        double distance = computeDistance(point_nearest, point_rand);
        
        if (distance <= adaptive_step) {
            candidate_point_[0] = point_rand[0];
            candidate_point_[1] = point_rand[1];
        } else {
            double ratio = adaptive_step / distance;
            candidate_point_[0] = point_nearest[0] + ratio * (point_rand[0] - point_nearest[0]);
            candidate_point_[1] = point_nearest[1] + ratio * (point_rand[1] - point_nearest[1]);
        }
        
        return candidate_point_;
    }   

    const std::vector<Node>& RRTPlanner::getTree() {

        return nodes_;
    }

    void RRTPlanner::setStart(double *start) {

        start_[0] = start[0];
        start_[1] = start[1];
    }

    void RRTPlanner::setGoal(double *goal) {

        goal_[0] = goal[0];
        goal_[1] = goal[1];
    }

// void RRTPlanner::smoothPath(std::vector<Node>& path) {
//         if (path.size() < 3) return;
        
//         int max_iterations = 50;
        
//         for (int iter = 0; iter < max_iterations; iter++) {
//             bool improved = false;
            
//             // Try to shortcut between nodes
//             for (size_t i = 0; i < path.size() - 1; i++) {
//                 for (size_t j = path.size() - 1; j > i + 1; j--) {
//                     // Check if we can connect node i directly to node j
//                     if (!collision_dect_.obstacleBetween(path[i].pos, path[j].pos)) {
//                         // Remove all nodes between i and j (the shortcut!)
//                         path.erase(path.begin() + i + 1, path.begin() + j);
//                         improved = true;
//                         break;
//                     }
//                 }
//                 if (improved) break;
//             }
            
//             // Stop if no improvements made
//             if (!improved) break;
//         }
//     }
    // std::vector<Node> RRTPlanner::extractPath() {
    //     std::vector<Node> path;
        
    //     // Find the goal node (last node added)
    //     int current_id = nodes_.size() - 1;
        
    //     // Backtrack from goal to start
    //     while (current_id != -1) {
    //         path.push_back(nodes_[current_id]);
    //         current_id = nodes_[current_id].parent_id;
    //     }
        
    //     // Reverse to get start-to-goal order
    //     std::reverse(path.begin(), path.end());
        
    //     return path;
    // }

};