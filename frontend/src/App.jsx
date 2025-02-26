import { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import './App.css';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function FilterOption({ name, value, options = {}, onChange }) {
  // Special handling for age ranges
  if (name === "Age") {
    const currentValue = value ? JSON.parse(value).value : [18, 100];
    return (
      <div className="filter-option">
        <label>{name}</label>
        <div className="age-range">
          <input
            type="number"
            min="18"
            max="100"
            value={currentValue[0]}
            onChange={(e) => {
              onChange(JSON.stringify({
                name: `${e.target.value}-${currentValue[1]}`,
                value: [parseInt(e.target.value), currentValue[1]]
              }));
            }}
          />
          <span>to</span>
          <input
            type="number"
            min="18"
            max="100"
            value={currentValue[1]}
            onChange={(e) => {
              onChange(JSON.stringify({
                name: `${currentValue[0]}-${e.target.value}`,
                value: [currentValue[0], parseInt(e.target.value)]
              }));
            }}
          />
        </div>
      </div>
    );
  }

  // Regular select for other filters
  return (
    <div className="filter-option">
      <label>{name}</label>
      <select value={value || ''} onChange={(e) => onChange(e.target.value)}>
        <option value="">Select...</option>
        {Object.entries(options).map(([optionName, optionValue]) => {
          const displayName = optionName;
          return (
            <option key={optionName} value={JSON.stringify({ name: optionName, value: optionValue })}>
              {displayName}
            </option>
          );
        })}
      </select>
    </div>
  );
}

FilterOption.propTypes = {
  name: PropTypes.string.isRequired,
  value: PropTypes.string.isRequired,
  options: PropTypes.object.isRequired,
  onChange: PropTypes.func.isRequired
};

function ExperimentRow({ experiment, filterOptions, onUpdate, onDelete }) {
  const addFilter = () => {
    if (experiment.filters.length < 3) {
      onUpdate({
        ...experiment,
        filters: [...experiment.filters, { key: '', value: '' }]
      });
    }
  };

  const updateFilter = (index, key, value, filterOptions) => {
    const newFilters = [...experiment.filters];
    if (!value) {
      var defaultValue;
      if (key === 'Age') {
        defaultValue = { name: '18-100', value: [18, 100] };
      } else{
        defaultValue = { name: Object.entries(filterOptions[key])[0][0], value: Object.entries(filterOptions[key])[0][1] };
      }
      console.log(key)
      console.log(filterOptions)
      console.log(defaultValue);
      newFilters[index] = { key, value: defaultValue };
    } else {
      try {
        const parsedValue = JSON.parse(value);
        console.log(parsedValue);
        newFilters[index] = { 
          key, 
          value: parsedValue
        };
      } catch (e) {
        console.error('Error parsing filter value:', e);
        newFilters[index] = { key, value: '' };
      }
    }
    onUpdate({
      ...experiment,
      filters: newFilters
    });
  };

  const removeFilter = (index) => {
    const newFilters = experiment.filters.filter((_, i) => i !== index);
    onUpdate({
      ...experiment,
      filters: newFilters
    });
  };

  return (
    <div className="experiment-row">
      <button className="delete-experiment" onClick={onDelete}>×</button>
      <div className="filters-container">
        {experiment.filters.map((filter, index) => (
          <div key={index} className="filter">
            <div className="filter-header">
              <select 
                value={filter.key} 
                onChange={(e) => updateFilter(index, e.target.value, '', filterOptions)}
              >
                <option value="">Select Filter</option>
                {Object.keys(filterOptions).map(key => (
                  <option key={key} value={key}>{key}</option>
                ))}
              </select>
              <button className="remove-filter" onClick={() => removeFilter(index)}>×</button>
            </div>
            {filter.key && (
              <FilterOption
                name={filter.key}
                value={filter.value ? JSON.stringify(filter.value) : ''}
                options={filterOptions[filter.key]}
                onChange={(value) => updateFilter(index, filter.key, value, filterOptions)}
              />
            )}
          </div>
        ))}
        {experiment.filters.length < 3 && (
          <button className="add-filter" onClick={addFilter}>+ Add Filter</button>
        )}
      </div>
    </div>
  );
}

ExperimentRow.propTypes = {
  experiment: PropTypes.shape({
    name: PropTypes.string.isRequired,
    filters: PropTypes.arrayOf(PropTypes.shape({
      key: PropTypes.string.isRequired,
      value: PropTypes.oneOfType([
        PropTypes.string,
        PropTypes.shape({
          name: PropTypes.string,
          value: PropTypes.oneOfType([
            PropTypes.string,
            PropTypes.number,
            PropTypes.array
          ])
        })
      ]).isRequired
    })).isRequired
  }).isRequired,
  filterOptions: PropTypes.object.isRequired,
  onUpdate: PropTypes.func.isRequired,
  onDelete: PropTypes.func.isRequired
};

function ResultsVisualization({ results }) {
  const probsData = {
    labels: Array.from({ length: 51 }, (_, i) => i.toString()),
    datasets: [
      ...results.map((result, index) => ({
        label: `${result.name} (n=${result.predictions.Count.reduce((a, b) => a + b, 0)})`,
        data: result.predictions.Probs,
        borderColor: `hsl(${index * 360/results.length}, 70%, 50%)`,
        tension: 0.1
      })),
      ...results.map((result, index) => ({
        label: `${result.name} (Actual)`,
        data: result.predictions["Data Probs"],
        borderColor: `hsl(${index * 360/results.length}, 70%, 50%)`,
        borderDash: [5, 5],
        hidden: results.length > 1,  // Only show if there's one experiment
        tension: 0.1
      }))
    ]
  };

  const lorenzData = {
    labels: Array.from({ length: 500 }, (_, i) => (i/499).toFixed(2)),
    datasets: results.map((result, index) => ({
      label: `${result.name} (n=${result.predictions.Count.reduce((a, b) => a + b, 0)})`,
      data: result.lorenz_points.y,
      borderColor: `hsl(${index * 360/results.length}, 70%, 50%)`,
      tension: 0.1
    })).concat([{
      label: 'y=x Line',
      data: Array.from({ length: 500 }, (_, i) => i / 499),
      borderColor: 'rgba(255, 99, 132, 1)',
      borderDash: [5, 5],
      tension: 0
    }])
  };

  const mixingData = {
    labels: results[0].mixing_points.x.map(x => x.toFixed(1)),
    datasets: results.map((result, index) => ({
      label: `${result.name} (n=${result.predictions.Count.reduce((a, b) => a + b, 0)})`,
      data: result.mixing_points.y,
      borderColor: `hsl(${index * 360/results.length}, 70%, 50%)`,
      tension: 0.1
    }))
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        onClick: (e, legendItem, legend) => {
          const index = legendItem.datasetIndex;
          const ci = legend.chart;
          ci.setDatasetVisibility(index, !ci.isDatasetVisible(index));
          ci.update();
        }
      },
      title: {
        display: true,
        font: {
          size: 16,
          weight: 'bold'
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
      }
    }
  };

  const probsChartOptions = {
    ...chartOptions,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Probability',
          font: {
            size: 14,
            weight: 'bold'
          }
        }
      },
      x: {
        title: {
          display: true,
          text: 'Body Count (50 means 50 or more)',
          font: {
            size: 14,
            weight: 'bold'
          }
        }
      }
    }
  };

  const lorenzChartOptions = {
    ...chartOptions,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Cumulative % of Total Body Counts',
          font: {
            size: 14,
            weight: 'bold'
          }
        }
      },
      x: {
        title: {
          display: true,
          text: 'Percentile of Population',
          font: {
            size: 14,
            weight: 'bold'
          }
        }
      }
    }
  };

  const mixingChartOptions = {
    ...chartOptions,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Probability',
          font: {
            size: 14,
            weight: 'bold'
          }
        }
      },
      x: {
        title: {
          display: true,
          text: 'λ (Mean Body Count Parameter)',
          font: {
            size: 14,
            weight: 'bold'
          }
        }
      }
    }
  };

  return (
    <div className="results-visualization">
      <div className="parameters-table">
        <h3>Parameters and Aggregate Statistics</h3>
        <table>
          <thead>
            <tr>
              <th>Experiment</th>
              <th>n</th>
              <th>θ</th>
              <th>α</th>
              <th>r</th>
              <th>Mean</th>
              <th>Median</th>
              <th>Mode</th>
              <th>p-value</th>
            </tr>
          </thead>
          <tbody>
            {results.map(result => (
              <tr key={result.name}>
                <td>{result.name}</td>
                <td>{result.predictions.Count.reduce((a, b) => a + b, 0)}</td>
                <td>{result.params[0].toFixed(3)}</td>
                <td>{result.params[1].toFixed(3)}</td>
                <td>{result.params[2].toFixed(3)}</td>
                <td>{result.aggregate_stats.mean.toFixed(2)}</td>
                <td>{result.aggregate_stats.median}</td>
                <td>{result.aggregate_stats.mode}</td>
                <td>{result.aggregate_stats.chi_square_p_value.toExponential(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="charts-container">
        <div className="chart">
          <h3>Probability Distribution</h3>
          <p className="chart-description">
            See the probability distribution of body counts for your selected demographics. 
            This shows how likely someone is to have a specific number of partners, 
            with 50+ representing having 50 or more partners.
          </p>
          <p className="chart-description">
            P.S. Click on the legend to hide/show the actual data.
          </p>
          <div className="chart-container">
            <Line data={probsData} options={probsChartOptions} />
          </div>
        </div>
        
        <div className="chart">
          <h3>Lorenz Curve</h3>
          <p className="chart-description">
            The Lorenz curve shows the inequality in body count distributions. 
            For example, if the curve shows (0.8, 0.2), it means the bottom 80% of people 
            contribute only 20% of the total body count. A perfectly equal distribution 
            would follow the diagonal line.
          </p>
          <div className="chart-container">
            <Line data={lorenzData} options={lorenzChartOptions} />
          </div>
        </div>
        
        <div className="chart">
          <h3>Mixing Distribution</h3>
          <p className="chart-description">
            This shows the underlying distribution of individual "sexual exposure parameters" (λ).
            Each person is assumed to have their own mean body count parameter λ, drawn from 
            this gamma distribution, which then determines their actual body count through 
            a Poisson process.
          </p>
          <div className="chart-container">
            <Line data={mixingData} options={mixingChartOptions} />
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [filterOptions, setFilterOptions] = useState({});
  const [experiments, setExperiments] = useState([]);
  const [results, setResults] = useState(null);

  useEffect(() => {
    fetch('/api/filter-options')
      .then(res => res.json())
      .then(data => setFilterOptions(data));
  }, []);

  useEffect(() => {
    document.title = "Partner Tally Pro"; // Set the tab name
  }, []);

  const addExperiment = () => {
    setExperiments([
      ...experiments,
      { 
        filters: experiments.length > 0 
          ? JSON.parse(JSON.stringify(experiments[experiments.length - 1].filters))  // Deep copy the filters from the last experiment
          : [] 
      }
    ]);
  };

  const updateExperiment = (index, updatedExperiment) => {
    const newExperiments = [...experiments];
    newExperiments[index] = updatedExperiment;
    setExperiments(newExperiments);
  };

  const deleteExperiment = (index) => {
    setExperiments(experiments.filter((_, i) => i !== index));
  };

  const generateExperimentName = (filters) => {
    if (filters.length === 0) {
        return "Entire Population";
    }
    return filters.map(filter => {
        if (filter.key && filter.value) { // Check if key and value are defined
            if (filter.key === "Age") {
                const [min, max] = filter.value.value;
                return `${filter.key}: ${min}-${max}`;
            }
            return `${filter.key}: ${filter.value.name}`;
        }
        return ""; // Return an empty string for undefined filters
    }).filter(name => name).join(", "); // Filter out empty names
  };

  const runExperiments = async () => {
    const payload = {
      experiments: experiments.map(exp => ({
        name: generateExperimentName(exp.filters),
        filters: Object.fromEntries(
          exp.filters.map(f => [f.key, f.value.value])
        )
      }))
    };

    try {
      const response = await fetch('/api/run-experiments', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error running experiments:', error);
    }
  };

  return (
    <div className="app">
      <div className="header">
        <h1 className="main-title">Partner Tally Pro 🔥</h1>
        <h2 className="subtitle">Your Guide to America's Body Count Stats</h2>
      </div>

      <div className="description-section">
        <p className="main-description">
          Ever wondered how your "number" stacks up? We've got the juicy details straight from 
          the CDC's 2022-23 National Survey of Family Growth. Slice and dice the data to see which groups 
          are really getting busy in the US! 🌶️ Compare different demographics and see who's 
          leading the scoreboard... for science, of course! 📊
        </p>
        <p className="technical-note">
          Using a spicy statistical cocktail: We fit a Negative Binomial Distribution with 
          Gamma-mixed parameters and a special spike at 1 (because sometimes that first 
          partner is extra special 😉).
        </p>
      </div>

      <div className="experiments-container">
        {experiments.map((experiment, index) => (
          <ExperimentRow
            key={index}
            experiment={experiment}
            filterOptions={filterOptions}
            onUpdate={(updated) => updateExperiment(index, updated)}
            onDelete={() => deleteExperiment(index)}
          />
        ))}
        <button className="add-experiment" onClick={addExperiment}>
          <span>+</span> Compare New Group
        </button>
      </div>

      {experiments.length > 0 && (
        <button className="run-experiments" onClick={runExperiments}>
          Run Experiments
        </button>
      )}

      {results && <ResultsVisualization results={results} />}
    </div>
  );
}

export default App; 