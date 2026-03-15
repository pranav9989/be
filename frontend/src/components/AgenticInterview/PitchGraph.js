import React from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

const PitchGraph = ({ pitchHistory, stabilityHistory, pitchTimestamps, livePitch }) => {
    // Calculate dynamic Y-axis range for pitch based on actual values
    const getPitchYAxisRange = () => {
        if (pitchHistory.length === 0) {
            return { min: 50, max: 400 };
        }

        const minPitch = Math.min(...pitchHistory);
        const maxPitch = Math.max(...pitchHistory);
        const padding = 20; // Add 20Hz padding

        return {
            min: Math.max(50, Math.floor(minPitch - padding)),
            max: Math.min(400, Math.ceil(maxPitch + padding))
        };
    };

    const pitchYRange = getPitchYAxisRange();

    // Create chart data with dual datasets
    const data = {
        labels: pitchTimestamps,
        datasets: [
            {
                label: 'Pitch (Hz)',
                data: pitchHistory,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.4,
                fill: true,
                pointRadius: 3,
                pointHoverRadius: 5,
                yAxisID: 'y-pitch', // Associate with left y-axis
            },
            // 🔥 FIXED: Stability history (not constant)
            {
                label: 'Stability (%)',
                data: stabilityHistory,
                borderColor: 'rgb(255, 159, 64)',
                backgroundColor: 'rgba(255, 159, 64, 0.1)',
                tension: 0.2,
                fill: false,
                pointRadius: 2,
                pointHoverRadius: 4,
                yAxisID: 'y-stability', // Associate with right y-axis
            },
            // Reference line for typical male range
            {
                label: 'Male Range (120Hz)',
                data: Array(pitchHistory.length).fill(120),
                borderColor: 'rgba(54, 162, 235, 0.3)',
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false,
                yAxisID: 'y-pitch',
            },
            // Reference line for typical female range
            {
                label: 'Female Range (200Hz)',
                data: Array(pitchHistory.length).fill(200),
                borderColor: 'rgba(255, 99, 132, 0.3)',
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false,
                yAxisID: 'y-pitch',
            }
        ],
    };

    // Chart options with dual axes
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 300 // Smooth updates
        },
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: {
                    usePointStyle: true,
                    boxWidth: 10,
                    font: {
                        size: 11
                    }
                }
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                callbacks: {
                    label: function (context) {
                        let label = context.dataset.label || '';
                        if (label) {
                            label += ': ';
                        }
                        if (context.dataset.label.includes('Pitch')) {
                            label += context.raw.toFixed(0) + ' Hz';
                        } else if (context.dataset.label.includes('Stability')) {
                            label += context.raw.toFixed(0) + '%';
                        } else {
                            label += context.raw.toFixed(0) + ' Hz';
                        }
                        return label;
                    }
                }
            },
        },
        scales: {
            'y-pitch': {
                type: 'linear',
                display: true,
                position: 'left',
                beginAtZero: false,
                min: pitchYRange.min,
                max: pitchYRange.max,
                title: {
                    display: true,
                    text: 'Frequency (Hz)'
                },
                grid: {
                    color: 'rgba(0, 0, 0, 0.05)',
                }
            },
            'y-stability': {
                type: 'linear',
                display: true,
                position: 'right',
                beginAtZero: true,
                min: 0,
                max: 100,
                title: {
                    display: true,
                    text: 'Stability (%)'
                },
                grid: {
                    drawOnChartArea: false, // Don't draw grid lines on this axis
                },
            },
            x: {
                display: true,
                title: {
                    display: true,
                    text: 'Time'
                },
                ticks: {
                    maxRotation: 45,
                    minRotation: 45
                }
            }
        },
    };

    // Calculate statistics and feedback
    const getPitchFeedback = () => {
        if (pitchHistory.length === 0) return null;

        const avg = pitchHistory.reduce((a, b) => a + b, 0) / pitchHistory.length;
        const max = Math.max(...pitchHistory);
        const min = Math.min(...pitchHistory);
        const stability = livePitch.stability;

        let feedback = [];
        let status = 'neutral';

        if (avg < 100) feedback.push("Deep voice range");
        else if (avg > 250) feedback.push("High voice range");
        else feedback.push("Normal voice range");

        if (stability > 80) {
            feedback.push("Very stable");
            status = 'excellent';
        } else if (stability > 60) {
            feedback.push("Moderately stable");
            status = 'good';
        } else if (stability > 40) {
            feedback.push("Variable");
            status = 'warning';
        } else {
            feedback.push("Highly variable");
            status = 'poor';
        }

        return { feedback: feedback.join(' · '), status, avg, max, min };
    };

    const stats = getPitchFeedback();

    return (
        <div className="pitch-analysis-container">
            <div className="pitch-graph-header">
                <h4>🎤 Live Pitch Analysis</h4>
                <div className="pitch-stats-badge">
                    <span className={`stability-indicator ${stats?.status || 'neutral'}`}>
                        {livePitch.stability.toFixed(0)}% Stable
                    </span>
                </div>
            </div>

            <div className="pitch-graph-wrapper">
                <Line data={data} options={options} />
            </div>

            {stats && (
                <div className="pitch-insights">
                    <div className="insight-row">
                        <span className="insight-label">Current:</span>
                        <span className="insight-value">{livePitch.mean.toFixed(0)} Hz</span>

                        <span className="insight-label">Range:</span>
                        <span className="insight-value">{livePitch.range.toFixed(0)} Hz</span>

                        <span className="insight-label">Avg:</span>
                        <span className="insight-value">{stats.avg.toFixed(0)} Hz</span>
                    </div>

                    <div className="insight-row">
                        <span className="insight-label">Peak:</span>
                        <span className="insight-value">{stats.max.toFixed(0)} Hz</span>

                        <span className="insight-label">Low:</span>
                        <span className="insight-value">{stats.min.toFixed(0)} Hz</span>

                        <span className="insight-label">Stability:</span>
                        <span className="insight-value">
                            <span className={`stability-badge ${stats.status}`}>
                                {livePitch.stability.toFixed(1)}%
                            </span>
                        </span>
                    </div>

                    <div className="feedback-message">
                        <span className={`feedback-badge ${stats.status}`}>●</span>
                        {stats.feedback}
                    </div>
                </div>
            )}
        </div>
    );
};

export default PitchGraph;