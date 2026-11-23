document.addEventListener('DOMContentLoaded', function () {
  const dataElement = document.getElementById('dashboard-data');
  if (!dataElement) return;
  if (typeof Chart === 'undefined') {
    const errorElement = document.getElementById('dashboardChartError');
    if (errorElement) {
      errorElement.textContent = 'Charts could not load. Check your internet connection and refresh the page.';
      errorElement.classList.remove('d-none');
    }
    return;
  }

  const data = JSON.parse(dataElement.textContent);
  const amber = '#f59e0b';
  const amberLight = '#fcd34d';
  const grid = '#e2e8f0';
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { displayColors: false }
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#64748b' } },
      y: { beginAtZero: true, grid: { color: grid }, ticks: { color: '#64748b' } }
    }
  };

  new Chart(document.getElementById('statusChart'), {
    type: 'bar',
    data: {
      labels: data.status_labels,
      datasets: [{ data: data.status_values, backgroundColor: [amberLight, amber], borderRadius: 6, maxBarThickness: 64 }]
    },
    options: {
      ...baseOptions,
      plugins: { ...baseOptions.plugins, tooltip: { callbacks: { label: item => `${item.formattedValue} records` } } },
      scales: { ...baseOptions.scales, y: { ...baseOptions.scales.y, title: { display: true, text: 'Records' } } }
    }
  });

  new Chart(document.getElementById('scoreChart'), {
    type: 'bar',
    data: {
      labels: data.score_labels,
      datasets: [{ data: data.score_values, backgroundColor: amber, borderRadius: 4, maxBarThickness: 42 }]
    },
    options: {
      ...baseOptions,
      plugins: { ...baseOptions.plugins, tooltip: { callbacks: { label: item => `${item.formattedValue} records` } } },
      scales: {
        ...baseOptions.scales,
        x: { ...baseOptions.scales.x, title: { display: true, text: 'Anomaly score interval' } },
        y: { ...baseOptions.scales.y, title: { display: true, text: 'Records' } }
      }
    }
  });

  const featureSelect = document.getElementById('featureSelect');
  const featureCanvas = document.getElementById('featureComparisonChart');
  if (featureSelect && featureCanvas) {
    const featureChart = new Chart(featureCanvas, {
      type: 'bar',
      data: {
        labels: data.status_labels,
        datasets: [{
          data: data.feature_stats[featureSelect.value] || [],
          backgroundColor: [amberLight, amber],
          borderRadius: 6,
          maxBarThickness: 72
        }]
      },
      options: {
        ...baseOptions,
        scales: {
          ...baseOptions.scales,
          y: { ...baseOptions.scales.y, beginAtZero: false, title: { display: true, text: featureSelect.value } }
        }
      }
    });

    featureSelect.addEventListener('change', function () {
      const feature = featureSelect.value;
      featureChart.data.datasets[0].data = data.feature_stats[feature] || [];
      featureChart.options.scales.y.title.text = feature;
      featureChart.update();
      document.getElementById('featureChartTitle').textContent =
        `Median ${feature} for normal and flagged rows, in the feature’s original units.`;
    });
  }

  const categoryCanvas = document.getElementById('categoryChart');
  if (categoryCanvas) {
    new Chart(categoryCanvas, {
      type: 'bar',
      data: {
        labels: data.category_labels,
        datasets: [{ data: data.category_values, backgroundColor: amber, borderRadius: 5, maxBarThickness: 48 }]
      },
      options: {
        ...baseOptions,
        indexAxis: 'y',
        scales: {
          x: { ...baseOptions.scales.y, max: 100, title: { display: true, text: 'Flagged records (%)' } },
          y: { ...baseOptions.scales.x }
        },
        plugins: {
          ...baseOptions.plugins,
          tooltip: { callbacks: { label: item => `${item.formattedValue}% flagged` } }
        }
      }
    });
  }

  const trendCanvas = document.getElementById('trendChart');
  if (trendCanvas) {
    new Chart(trendCanvas, {
      type: 'bar',
      data: {
        labels: data.trend_labels,
        datasets: [{ data: data.trend_values, backgroundColor: amber, borderRadius: 4, maxBarThickness: 36 }]
      },
      options: {
        ...baseOptions,
        scales: {
          ...baseOptions.scales,
          y: { ...baseOptions.scales.y, max: 100, title: { display: true, text: 'Flagged records (%)' } }
        },
        plugins: {
          ...baseOptions.plugins,
          tooltip: { callbacks: { label: item => `${item.formattedValue}% flagged` } }
        }
      }
    });
  }
});
