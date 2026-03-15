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

// Register ChartJS components (removed TimeScale)
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

const PitchGraph = ({ pitchHistory = [], stabilityHistory = [], pitchTimestamps = [], livePitch }) => {
    // Limit to last 20 points for better visibility
    const maxPoints = 20;
    const recentPitch = pitchHistory.slice(-maxPoints);
    const recentStability = stabilityHistory.slice(-maxPoints);
    const recentTimestamps = pitchTimestamps.slice(-maxPoints);

    // Calculate dynamic Y-axis range for pitch
    const getPitchYAxisRange = () => {
        if (recentPitch.length === 0) {
            return { min: 50, max: 400 };
        }

        const validPitches = recentPitch.filter(p => p > 0 && p < 500);
        if (validPitches.length === 0) return { min: 50, max: 400 };

        const minPitch = Math.min(...validPitches);
        const maxPitch = Math.max(...validPitches);
        const padding = 20;

        return {
            min: Math.max(50, Math.floor(minPitch - padding)),
            max: Math.min(400, Math.ceil(maxPitch + padding))
        };
    };

    const pitchYRange = getPitchYAxisRange();

    // Create chart data with both datasets
    const data = {
        labels: recentTimestamps,
        datasets: [
            {
                label: 'Pitch (Hz)',
                data: recentPitch,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.4,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6,
                borderWidth: 2,
                yAxisID: 'y-pitch',
            },
            {
                label: 'Stability (%)',
                data: recentStability,
                borderColor: 'rgb(255, 159, 64)',
                backgroundColor: 'rgba(255, 159, 64, 0.1)',
                tension: 0.2,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6,
                borderWidth: 2,
                yAxisID: 'y-stability',
            },
            // Reference line for typical male range
            {
                label: 'Male Range (120Hz)',
                data: Array(recentTimestamps.length).fill(120),
                borderColor: 'rgba(54, 162, 235, 0.3)',
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false,
                borderWidth: 1,
                yAxisID: 'y-pitch',
            },
            // Reference line for typical female range
            {
                label: 'Female Range (200Hz)',
                data: Array(recentTimestamps.length).fill(200),
                borderColor: 'rgba(255, 99, 132, 0.3)',
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false,
                borderWidth: 1,
                yAxisID: 'y-pitch',
            }
        ],
    };

    // Chart options with improved layout
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 300
        },
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                display: true,
                position: 'top',
                align: 'center',
                labels: {
                    usePointStyle: true,
                    boxWidth: 8,
                    padding: 15,
                    font: {
                        size: 11,
                        weight: '500'
                    }
                }
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleFont: { size: 12 },
                bodyFont: { size: 11 },
                padding: 8,
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
                    text: 'Frequency (Hz)',
                    font: { size: 10, weight: '500' }
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)',
                },
                ticks: {
                    stepSize: 50,
                    font: { size: 9 }
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
                    text: 'Stability (%)',
                    font: { size: 10, weight: '500' }
                },
                grid: {
                    drawOnChartArea: false,
                },
                ticks: {
                    stepSize: 20,
                    font: { size: 9 },
                    callback: function (value) {
                        return value + '%';
                    }
                }
            },
            x: {
                display: true,
                title: {
                    display: true,
                    text: 'Time',
                    font: { size: 10, weight: '500' }
                },
                ticks: {
                    maxRotation: 30,
                    minRotation: 30,
                    font: { size: 8 },
                    maxTicksLimit: 8
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                }
            }
        },
    };

    // Calculate statistics
    const getPitchFeedback = () => {
        if (recentPitch.length === 0) return null;

        const validPitches = recentPitch.filter(p => p > 0);
        if (validPitches.length === 0) return null;

        const avg = validPitches.reduce((a, b) => a + b, 0) / validPitches.length;
        const max = Math.max(...validPitches);
        const min = Math.min(...validPitches);
        const stability = livePitch?.stability ||
            (recentStability.length > 0 ? recentStability[recentStability.length - 1] : 50);

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

        return { feedback: feedback.join(' · '), status, avg, max, min, stability };
    };

    const stats = getPitchFeedback();

    return (
        <div className="pitch-analysis-container">
            <div className="pitch-graph-header" style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '10px'
            }}>
                <h4 style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>🎤 Live Voice Analysis</h4>
                <div className="pitch-stats-badge">
                    <span className={`stability-indicator ${stats?.status || 'neutral'}`} style={{
                        padding: '2px 8px',
                        borderRadius: '12px',
                        fontSize: '0.75rem',
                        background: stats?.status === 'excellent' ? 'rgba(74,222,128,0.2)' :
                            stats?.status === 'good' ? 'rgba(74,222,128,0.15)' :
                                stats?.status === 'warning' ? 'rgba(255,159,64,0.2)' :
                                    stats?.status === 'poor' ? 'rgba(239,68,68,0.2)' :
                                        'rgba(255,255,255,0.1)',
                        color: stats?.status === 'excellent' ? '#4ADE80' :
                            stats?.status === 'good' ? '#4ADE80' :
                                stats?.status === 'warning' ? '#FF9F40' :
                                    stats?.status === 'poor' ? '#EF4444' :
                                        'var(--text-muted)'
                    }}>
                        {stats?.stability?.toFixed(0) || '0'}% Stable
                    </span>
                </div>
            </div>

            <div className="pitch-graph-wrapper" style={{ height: '180px', marginBottom: '10px' }}>
                <Line data={data} options={options} />
            </div>

            {stats && (
                <div className="pitch-insights" style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '6px',
                    fontSize: '0.8rem',
                    background: 'var(--bg-card)',
                    padding: '8px 10px',
                    borderRadius: '8px',
                    border: '1px solid var(--border)'
                }}>
                    <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                        <div><span style={{ color: 'var(--text-muted)' }}>Current:</span> <strong style={{ color: 'rgb(75, 192, 192)' }}>{livePitch?.mean?.toFixed(0) || '0'} Hz</strong></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Range:</span> <strong>{livePitch?.range?.toFixed(0) || '0'} Hz</strong></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Avg:</span> <strong>{stats.avg.toFixed(0)} Hz</strong></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Peak:</span> <strong>{stats.max.toFixed(0)} Hz</strong></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Low:</span> <strong>{stats.min.toFixed(0)} Hz</strong></div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '2px' }}>
                        <span style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            background: stats.status === 'excellent' ? '#4ADE80' :
                                stats.status === 'good' ? '#4ADE80' :
                                    stats.status === 'warning' ? '#FF9F40' :
                                        stats.status === 'poor' ? '#EF4444' :
                                            'var(--text-muted)'
                        }} />
                        <span style={{ color: 'var(--text-secondary)' }}>{stats.feedback}</span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PitchGraph;