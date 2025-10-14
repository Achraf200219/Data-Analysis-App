import React, { useState, useRef, useEffect } from 'react';import React, { useState } from 'react';

import Plot from 'react-plotly.js';import Plot from 'react-plotly.js';

import { ThemeToggle } from './components/ThemeToggle';

import './App.css';/**

 * Top‑level React component implementing a user interface to Vanna.

/** *

 * Data Analysis Platform - Multi-format data analysis with file uploads. * The layout loosely follows the Streamlit example in the official Vanna

 */ * repository: a sidebar with toggles controls whether certain outputs (SQL,

export default function App() { * table, plotly code, chart, summary, follow‑ups) are shown【302593729442208†L16-L24】,

  const [dataSources, setDataSources] = useState({}); * a button fetches suggested questions from the backend, an input box accepts

  const [activeSource, setActiveSource] = useState(null); * arbitrary questions and, upon submission, a sequence of API calls

  const [messages, setMessages] = useState([]); * generates the SQL, runs it, creates a chart, summarises the results and

  const [currentMessage, setCurrentMessage] = useState(''); * suggests follow‑up questions.  Each of these pieces of data is stored in

  const [currentResult, setCurrentResult] = useState(null); * React state and conditionally rendered based on the toggles.

  const [suggestions, setSuggestions] = useState([]); */

  const [uploadLoading, setUploadLoading] = useState(false);export default function App() {

  const [chatLoading, setChatLoading] = useState(false);  // Output settings toggles (mirrors Streamlit sidebar checkboxes)

    const [settings, setSettings] = useState({

  const [settings, setSettings] = useState({    showSQL: true,

    showQuery: true,    showTable: true,

    showTable: true,    showPlotlyCode: true,

    showChart: true,    showChart: true,

    showSummary: true,    showSummary: true,

    showFollowup: true,    showFollowup: true,

  });  });



  const fileInputRef = useRef(null);  // Suggested questions returned from the backend

  const [suggestions, setSuggestions] = useState([]);

  useEffect(() => {  // The current question typed or clicked by the user

    loadDataSources();  const [currentQuestion, setCurrentQuestion] = useState('');

  }, []);  // Whether a request is currently in flight

  const [loading, setLoading] = useState(false);

  useEffect(() => {  // Results of the last query (SQL, rows, chart, etc.)

    if (activeSource) {  const [result, setResult] = useState(null);

      fetchSuggestions();  // Error message, if any

    }  const [error, setError] = useState(null);

  }, [activeSource]);

  /**

  const loadDataSources = async () => {   * Update a single setting in the settings state.

    try {   * @param {string} key

      const response = await fetch('/api/data-sources');   */

      const data = await response.json();  const toggleSetting = (key) => {

      setDataSources(data.sources);    setSettings((prev) => ({ ...prev, [key]: !prev[key] }));

      setActiveSource(data.active_source);  };

    } catch (error) {

      console.error('Failed to load data sources:', error);  /**

    }   * Request suggested questions from the backend.

  };   */

  const fetchSuggestions = async () => {

  const handleFileUpload = async (event) => {    try {

    const files = event.target.files;      const resp = await fetch('/api/generate-questions');

    if (!files || files.length === 0) return;      const data = await resp.json();

      setSuggestions(data.questions || []);

    setUploadLoading(true);    } catch (e) {

      console.error(e);

    for (const file of files) {      setError('Failed to fetch suggested questions');

      try {    }

        const formData = new FormData();  };

        formData.append('file', file);

  /**

        const response = await fetch('/api/upload-file', {   * Reset all UI state.  Clears the current question, result and error.

          method: 'POST',   */

          body: formData,  const reset = () => {

        });    setCurrentQuestion('');

    setResult(null);

        if (!response.ok) {    setError(null);

          let errorMessage = 'Upload failed';  };

          try {

            const error = await response.json();  /**

            errorMessage = error.detail || 'Upload failed';   * Execute the full question pipeline: generate SQL, run SQL, determine

          } catch {   * whether to draw a chart, draw the chart, summarise and suggest follow‑ups.

            const text = await response.text();   * This method mirrors the sequence of calls in the Streamlit demo【302593729442208†L61-L152】.

            errorMessage = text || `HTTP ${response.status} ${response.statusText}`;   *

          }   * @param {string} question The natural language query to ask Vanna.

          throw new Error(errorMessage);   */

        }  const askQuestion = async (question) => {

    if (!question) return;

        const fileInfo = await response.json();    setLoading(true);

            setError(null);

        setDataSources(prev => ({    setResult(null);

          ...prev,    try {

          [fileInfo.file_id]: fileInfo      // 1. Generate SQL from the question

        }));      const sqlResp = await fetch(

        `/api/generate-sql?question=${encodeURIComponent(question)}`,

        if (!activeSource) {      );

          setActiveSource(fileInfo.file_id);      const sqlData = await sqlResp.json();

        }      if (!sqlResp.ok) throw new Error(sqlData.detail || 'Error generating SQL');

      const sql = sqlData.sql;

        addMessage('system', `Successfully loaded ${fileInfo.name} (${fileInfo.type})`);

              // 2. Run the SQL and get a subset of the dataframe

      } catch (error) {      const runResp = await fetch('/api/run-sql', {

        addMessage('error', `Failed to upload ${file.name}: ${error.message}`);        method: 'POST',

      }        headers: { 'Content-Type': 'application/json' },

    }        body: JSON.stringify({ sql }),

      });

    setUploadLoading(false);      const runData = await runResp.json();

    if (fileInputRef.current) {      if (!runResp.ok) throw new Error(runData.detail || 'Error running SQL');

      fileInputRef.current.value = '';      const df = runData.df || [];

    }

  };      // 3. Decide whether we should generate a chart

      let shouldChart = false;

  const switchDataSource = async (sourceId) => {      if (settings.showChart) {

    try {        const shouldResp = await fetch('/api/should-generate-chart', {

      setActiveSource(sourceId);          method: 'POST',

      const formData = new FormData();          headers: { 'Content-Type': 'application/json' },

      formData.append('source_id', sourceId);          body: JSON.stringify({ df }),

              });

      const response = await fetch('/api/set-active-source', {        const shouldData = await shouldResp.json();

        method: 'POST',        shouldChart = shouldData.should_generate_chart;

        body: formData,      }

      });

            // 4. Generate Plotly code and figure if appropriate

      if (response.ok) {      let plotlyCode = '';

        addMessage('system', `Switched to data source: ${dataSources[sourceId]?.name || sourceId}`);      let figJson = null;

        setCurrentResult(null);      if (shouldChart && settings.showChart) {

      }        const plotResp = await fetch('/api/generate-plot', {

    } catch (error) {          method: 'POST',

      addMessage('error', `Failed to switch data source: ${error.message}`);          headers: { 'Content-Type': 'application/json' },

    }          body: JSON.stringify({ question, sql, df }),

  };        });

        const plotData = await plotResp.json();

  const deleteDataSource = async (sourceId) => {        plotlyCode = plotData.plotly_code || '';

    if (!confirm(`Are you sure you want to delete "${dataSources[sourceId]?.name}"?`)) {        figJson = plotData.fig || null;

      return;      }

    }

      // 5. Generate summary

    try {      let summary = null;

      const response = await fetch(`/api/remove-source/${sourceId}`, {      if (settings.showSummary) {

        method: 'DELETE',        const summaryResp = await fetch('/api/generate-summary', {

      });          method: 'POST',

          headers: { 'Content-Type': 'application/json' },

      if (!response.ok) {          body: JSON.stringify({ question, df }),

        let errorMessage = 'Delete failed';        });

        try {        const summaryData = await summaryResp.json();

          const error = await response.json();        summary = summaryData.summary || null;

          errorMessage = error.detail || 'Delete failed';      }

        } catch {

          errorMessage = `HTTP ${response.status} ${response.statusText}`;      // 6. Generate follow‑up questions

        }      let followups = [];

        throw new Error(errorMessage);      if (settings.showFollowup) {

      }        const followResp = await fetch('/api/generate-followup', {

          method: 'POST',

      setDataSources(prev => {          headers: { 'Content-Type': 'application/json' },

        const newSources = { ...prev };          body: JSON.stringify({ question, sql, df }),

        delete newSources[sourceId];        });

        return newSources;        const followData = await followResp.json();

      });        followups = followData.followup_questions || [];

      }

      if (activeSource === sourceId) {

        setActiveSource(null);      // Update result state

        setCurrentResult(null);      setResult({ sql, df, plotlyCode, figJson, summary, followups });

        addMessage('system', 'Active data source deleted. Please select a new data source.');    } catch (e) {

      } else {      console.error(e);

        addMessage('system', `Successfully deleted data source: ${dataSources[sourceId]?.name}`);      setError(e.message || 'An error occurred');

      }    } finally {

      setLoading(false);

    } catch (error) {    }

      addMessage('error', `Failed to delete data source: ${error.message}`);  };

    }

  };  /**

   * Handler for submitting the input box.

  const fetchSuggestions = async () => {   */

    try {  const handleSubmit = async (e) => {

      const response = await fetch('/api/generate-questions');    e.preventDefault();

      if (response.ok) {    if (!currentQuestion.trim()) return;

        const data = await response.json();    const q = currentQuestion.trim();

        setSuggestions(data.questions || []);    setCurrentQuestion('');

      }    await askQuestion(q);

    } catch (error) {  };

      console.error('Failed to fetch suggestions:', error);

    }  /**

  };   * Handler for clicking a suggested or follow‑up question.

   * Immediately asks the question and clears the suggestions list.

  const addMessage = (role, content, result = null) => {   *

    setMessages(prev => [...prev, {   * @param {string} q

      role,   */

      content,  const handleQuestionClick = async (q) => {

      result,    setSuggestions([]);

      timestamp: new Date().toLocaleTimeString()    await askQuestion(q);

    }]);  };

  };

  return (

  const handleSendMessage = async () => {    <div style={{ display: 'flex', flex: 1 }}>

    if (!currentMessage.trim() || chatLoading) return;      {/* Sidebar settings */}

          <aside

    const messageText = currentMessage.trim();        style={{

    setCurrentMessage('');          width: '220px',

    addMessage('user', messageText);          padding: '1rem',

    setChatLoading(true);          borderRight: '1px solid #ddd',

          backgroundColor: '#f5f5f5',

    try {          boxSizing: 'border-box',

      const response = await fetch('/api/chat', {        }}

        method: 'POST',      >

        headers: { 'Content-Type': 'application/json' },        <h2 style={{ marginTop: 0 }}>Output Settings</h2>

        body: JSON.stringify({         {Object.keys(settings).map((key) => (

          message: messageText,          <label

          source_id: activeSource             key={key}

        }),            style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}

      });          >

            <input

      if (!response.ok) throw new Error(`HTTP ${response.status}`);              type="checkbox"

              checked={settings[key]}

      const data = await response.json();              onChange={() => toggleSetting(key)}

      let enhancedResult = { ...data };              style={{ marginRight: '0.5rem' }}

                  />

      if (data.results && data.results.length > 0) {            {key.replace(/([A-Z])/g, ' $1')}

        try {          </label>

          const queryInfo = {        ))}

            query_type: data.query_type,        <button

            query: data.query,          onClick={reset}

            data_source: data.data_source,          style={{ marginTop: '1rem', padding: '0.5rem 0.75rem', width: '100%' }}

            source_type: data.source_type        >

          };          Reset

        </button>

          if (settings.showChart) {      </aside>

            try {

              const plotResponse = await fetch('/api/generate-plot', {      {/* Main content */}

                method: 'POST',      <main style={{ flex: 1, padding: '1rem', overflowY: 'auto' }}>

                headers: { 'Content-Type': 'application/json' },        <h1 style={{ marginTop: 0 }}>Vanna AI (React)</h1>

                body: JSON.stringify({

                  question: messageText,        {/* Suggestion button */}

                  query_info: queryInfo,        <div style={{ marginBottom: '1rem' }}>

                  df: data.results,          <button

                  source_id: activeSource            onClick={fetchSuggestions}

                }),            style={{ padding: '0.5rem 1rem' }}

              });            disabled={loading}

              if (plotResponse.ok) {          >

                const plotData = await plotResponse.json();            {suggestions.length > 0 ? 'Refresh Suggested Questions' : 'Show Suggested Questions'}

                enhancedResult.plot_data = plotData.fig ? JSON.parse(plotData.fig) : null;          </button>

                enhancedResult.plotly_code = plotData.plotly_code;        </div>

                enhancedResult.chart_error = plotData.error || null;        {/* Render suggested questions */}

              }        {suggestions.length > 0 && (

            } catch (error) {          <div style={{ marginBottom: '1rem' }}>

              enhancedResult.chart_error = 'Failed to generate chart';            <p><strong>Suggested Questions:</strong></p>

            }            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>

          }              {suggestions.map((q) => (

                <button

          if (settings.showSummary) {                  key={q}

            try {                  onClick={() => handleQuestionClick(q)}

              const summaryResponse = await fetch('/api/generate-summary', {                  style={{ padding: '0.25rem 0.5rem', cursor: 'pointer' }}

                method: 'POST',                >

                headers: { 'Content-Type': 'application/json' },                  {q}

                body: JSON.stringify({                </button>

                  question: messageText,              ))}

                  query_info: queryInfo,            </div>

                  df: data.results,          </div>

                  source_id: activeSource        )}

                }),

              });        {/* Input form */}

              if (summaryResponse.ok) {        <form onSubmit={handleSubmit} style={{ marginBottom: '1rem' }}>

                const summaryData = await summaryResponse.json();          <input

                enhancedResult.summary = summaryData.summary;            type="text"

              }            placeholder="Ask me a question about your data"

            } catch (error) {            value={currentQuestion}

              enhancedResult.summary_error = 'Failed to generate summary';            onChange={(e) => setCurrentQuestion(e.target.value)}

            }            style={{ flexGrow: 1, padding: '0.5rem', width: '70%' }}

          }          />

          <button type="submit" style={{ padding: '0.5rem 1rem', marginLeft: '0.5rem' }} disabled={loading}>

          if (settings.showFollowup) {            {loading ? 'Loading...' : 'Ask'}

            try {          </button>

              const followupResponse = await fetch('/api/generate-followup', {        </form>

                method: 'POST',

                headers: { 'Content-Type': 'application/json' },        {/* Error message */}

                body: JSON.stringify({        {error && (

                  question: messageText,          <div style={{ color: 'red', marginBottom: '1rem' }}>{error}</div>

                  query_info: queryInfo,        )}

                  df: data.results,

                  source_id: activeSource        {/* Render results */}

                }),        {result && (

              });          <div style={{ marginBottom: '2rem' }}>

              if (followupResponse.ok) {            {/* SQL output */}

                const followupData = await followupResponse.json();            {settings.showSQL && result.sql && (

                enhancedResult.followup_questions = followupData.followup_questions;              <div style={{ marginBottom: '1rem' }}>

              }                <h3>Generated SQL</h3>

            } catch (error) {                <pre

              enhancedResult.followup_error = 'Failed to generate follow-up questions';                  style={{

            }                    backgroundColor: '#f0f0f0',

          }                    padding: '0.75rem',

        } catch (additionalError) {                    overflowX: 'auto',

          console.warn('Failed to generate additional content:', additionalError);                  }}

        }                >

      }                  {result.sql}

                      </pre>

      setCurrentResult(enhancedResult);              </div>

      addMessage('assistant', 'Here are the results:', enhancedResult);            )}

            {/* Table output */}

    } catch (error) {            {settings.showTable && result.df && result.df.length > 0 && (

      addMessage('error', `Failed to process query: ${error.message}`);              <div style={{ marginBottom: '1rem' }}>

    } finally {                <h3>Top Rows</h3>

      setChatLoading(false);                <table style={{ borderCollapse: 'collapse', width: '100%' }}>

    }                  <thead>

  };                    <tr>

                      {Object.keys(result.df[0]).map((col) => (

  const handleSuggestionClick = (suggestion) => {                        <th

    setCurrentMessage(suggestion);                          key={col}

  };                          style={{ border: '1px solid #ddd', padding: '0.5rem', textAlign: 'left' }}

                        >

  const toggleSetting = (key) => {                          {col}

    setSettings(prev => ({                        </th>

      ...prev,                      ))}

      [key]: !prev[key]                    </tr>

    }));                  </thead>

  };                  <tbody>

                    {result.df.map((row, idx) => (

  return (                      <tr key={idx}>

    <div className="app">                        {Object.values(row).map((cell, i) => (

      <nav className="navbar">                          <td

        <div className="nav-brand">                            key={i}

          <span className="brand-text">Data Analysis Platform</span>                            style={{ border: '1px solid #ddd', padding: '0.5rem' }}

        </div>                          >

                                    {String(cell)}

        <div className="nav-actions">                          </td>

          <div className="upload-section">                        ))}

            <input                      </tr>

              type="file"                    ))}

              ref={fileInputRef}                  </tbody>

              onChange={handleFileUpload}                </table>

              multiple              </div>

              accept=".csv,.json,.xlsx,.parquet,.db,.sqlite"            )}

              style={{ display: 'none' }}            {/* Plotly code */}

            />            {settings.showPlotlyCode && result.plotlyCode && result.plotlyCode.trim() !== '' && (

            <button              <div style={{ marginBottom: '1rem' }}>

              onClick={() => fileInputRef.current?.click()}                <h3>Plotly Python Code</h3>

              disabled={uploadLoading}                <pre

              className="upload-btn"                  style={{

            >                    backgroundColor: '#f0f0f0',

              {uploadLoading ? 'Uploading...' : 'Upload Files'}                    padding: '0.75rem',

            </button>                    overflowX: 'auto',

          </div>                  }}

                >

          <div className="source-selector">                  {result.plotlyCode}

            <label>Data Sources:</label>                </pre>

            <div className="source-list">              </div>

              {Object.keys(dataSources).length === 0 ? (            )}

                <div className="no-sources">No data sources uploaded</div>            {/* Chart */}

              ) : (            {settings.showChart && result.figJson && (

                Object.entries(dataSources).map(([id, source]) => (              <div style={{ marginBottom: '1rem' }}>

                  <div key={id} className={`source-item ${activeSource === id ? 'active' : ''}`}>                <h3>Chart</h3>

                    <div className="source-info" onClick={() => switchDataSource(id)}>                <Plot

                      <span className="source-name">{source.name}</span>                  data={JSON.parse(result.figJson).data}

                      <span className="source-type">({source.type})</span>                  layout={JSON.parse(result.figJson).layout}

                    </div>                  style={{ width: '100%', height: '100%' }}

                    <button                  useResizeHandler

                      className="delete-btn"                />

                      onClick={(e) => {              </div>

                        e.stopPropagation();            )}

                        deleteDataSource(id);            {/* Summary */}

                      }}            {settings.showSummary && result.summary && (

                      title="Delete this data source"              <div style={{ marginBottom: '1rem' }}>

                    >                <h3>Summary</h3>

                      ×                <p>{result.summary}</p>

                    </button>              </div>

                  </div>            )}

                ))            {/* Follow‑up questions */}

              )}            {settings.showFollowup && result.followups && result.followups.length > 0 && (

            </div>              <div style={{ marginBottom: '1rem' }}>

          </div>                <h3>Follow‑up Questions</h3>

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>

          <div className="settings-panel">                  {result.followups.slice(0, 5).map((q) => (

            <span>Show:</span>                    <button

            {Object.entries({                      key={q}

              showQuery: 'SQL',                      onClick={() => handleQuestionClick(q)}

              showChart: 'Chart',                      style={{ padding: '0.25rem 0.5rem', cursor: 'pointer' }}

              showTable: 'Table',                    >

              showSummary: 'Summary',                      {q}

              showFollowup: 'Follow-up'                    </button>

            }).map(([key, label]) => (                  ))}

              <label key={key} className="setting-toggle">                </div>

                <input              </div>

                  type="checkbox"            )}

                  checked={settings[key]}          </div>

                  onChange={() => toggleSetting(key)}        )}

                />      </main>

                <span>{label}</span>    </div>

              </label>  );

            ))}}
          </div>

          <ThemeToggle className="theme-toggle-nav" />
        </div>
      </nav>

      <main className="main-content">
        <div className="chat-panel">
          <div className="chat-header">
            <h3>Chat Interface</h3>
            <div className="source-info">
              {activeSource ? (
                <span>Active: {dataSources[activeSource]?.name}</span>
              ) : (
                <span>No data source selected</span>
              )}
            </div>
          </div>

          <div className="messages-container">
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                <div className="message-header">
                  <span className="role-icon">
                    {message.role === 'user' ? 'User' : 
                     message.role === 'assistant' ? 'Assistant' : 
                     message.role === 'error' ? 'Error' : 'System'}
                  </span>
                  <span className="timestamp">{message.timestamp}</span>
                </div>
                <div className="message-content">{message.content}</div>
              </div>
            ))}
            
            {chatLoading && (
              <div className="message assistant">
                <div className="message-header">
                  <span className="role-icon">Assistant</span>
                  <span className="timestamp">Processing...</span>
                </div>
                <div className="message-content">
                  <div className="loading-spinner">Analyzing your query...</div>
                </div>
              </div>
            )}
          </div>

          <div className="chat-input">
            <div className="suggestions">
              {suggestions.length > 0 ? (
                suggestions.slice(0, 3).map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="suggestion-btn"
                  >
                    {suggestion}
                  </button>
                ))
              ) : (
                <div className="no-suggestions">
                  {activeSource ? 'Loading suggested questions...' : 'Select a data source to see suggested questions'}
                </div>
              )}
            </div>
            
            <div className="input-row">
              <input
                type="text"
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Ask a question about your data..."
                disabled={chatLoading || !activeSource}
              />
              <button
                onClick={handleSendMessage}
                disabled={chatLoading || !activeSource || !currentMessage.trim()}
                className="send-btn"
              >
                {chatLoading ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        </div>

        <div className="panel-divider" title="Drag to resize panels"></div>

        <div className="results-panel">
          <div className="results-header">
            <h3>Analysis Results</h3>
          </div>

          <div className="results-content">
            {currentResult ? (
              <>
                {settings.showQuery && currentResult.query && (
                  <div className="result-section">
                    <h4>Generated {currentResult.query_type.toUpperCase()} Query</h4>
                    <pre className="code-block">{currentResult.query}</pre>
                  </div>
                )}

                {settings.showChart && currentResult && (
                  <div className="result-section">
                    <h4>Visualization</h4>
                    <div className="chart-container">
                      {currentResult.chart_error ? (
                        <div className="chart-error">
                          Error: {currentResult.chart_error}
                        </div>
                      ) : currentResult.plot_data ? (
                        <Plot
                          data={currentResult.plot_data.data}
                          layout={{
                            ...currentResult.plot_data.layout,
                            autosize: true,
                            responsive: true
                          }}
                          config={{ responsive: true }}
                          style={{ width: '100%', height: '400px' }}
                        />
                      ) : (
                        <div className="chart-placeholder">
                          {currentResult.results && currentResult.results.length > 0 
                            ? "Generating chart..."
                            : "Chart will appear when data is available"
                          }
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {settings.showTable && currentResult.results && currentResult.results.length > 0 && (
                  <div className="result-section">
                    <h4>Data Table ({currentResult.total_rows} total rows)</h4>
                    <div className="table-container">
                      <table className="results-table">
                        <thead>
                          <tr>
                            {currentResult.results[0] && Object.keys(currentResult.results[0]).map(col => (
                              <th key={col}>{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {currentResult.results.slice(0, 100).map((row, index) => (
                            <tr key={index}>
                              {Object.values(row).map((value, i) => (
                                <td key={i}>{value}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {currentResult.results.length > 100 && (
                        <div className="table-note">
                          Showing first 100 rows of {currentResult.total_rows} total rows
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {settings.showSummary && currentResult && (
                  <div className="result-section">
                    <h4>Summary</h4>
                    <div className="summary-content">
                      {currentResult.summary_error ? (
                        <div className="summary-error">
                          Error: {currentResult.summary_error}
                        </div>
                      ) : currentResult.summary ? (
                        currentResult.summary
                      ) : (
                        currentResult.results && currentResult.results.length > 0 
                          ? "Generating summary..."
                          : "Summary will appear when data is available"
                      )}
                    </div>
                  </div>
                )}

                {settings.showFollowup && currentResult && (
                  <div className="result-section">
                    <h4>Follow-up Questions</h4>
                    <div className="followup-questions">
                      {currentResult.followup_error ? (
                        <div className="followup-error">
                          Error: {currentResult.followup_error}
                        </div>
                      ) : currentResult.followup_questions && currentResult.followup_questions.length > 0 ? (
                        currentResult.followup_questions.map((question, index) => (
                          <button
                            key={index}
                            onClick={() => handleSuggestionClick(question)}
                            className="followup-btn"
                          >
                            {question}
                          </button>
                        ))
                      ) : (
                        <div className="no-followup">
                          {currentResult.results && currentResult.results.length > 0 
                            ? "Generating follow-up questions..."
                            : "Follow-up questions will appear when data is available"
                          }
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="results-placeholder">
                <div className="placeholder-content">
                  <h3>Ready for Analysis</h3>
                  <p>
                    {!activeSource 
                      ? "Upload a data file and ask questions to see results here" 
                      : "Ask a question about your data to see analysis results"}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
