"""
@author: Moussa Kalla
"""

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "LinearRegression.h"

std::pair<std::vector<std::vector<double>>, std::vector<double>> loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value, label;

        while (ss >> value) {
            if (ss.peek() == ',') ss.ignore();
            row.push_back(value);
        }
        label = row.back();
        row.pop_back();
        X.push_back(row);
        y.push_back(label);
    }

    return {X, y};
}

int main() {
    auto [X, y] = loadCSV("../data/dataset.csv");

    LinearRegression model(0.01, 1000);
    model.fit(X, y);

    std::cout << "Model evaluation (RMSE): " << model.evaluate(X, y) << std::endl;

    std::vector<std::vector<double>> test_data = {{2100, 3}, {2500, 4}};
    auto predictions = model.predict(test_data);

    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Prediction for sample " << i + 1 << ": " << predictions[i] << std::endl;
    }

    return 0;
}
