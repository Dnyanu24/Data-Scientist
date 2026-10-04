import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { projectsApi } from '../api/client';

export default function DatasetsPage() {
  const [projects, setProjects] = useState<any[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    projectsApi.list().then((r) => setProjects(r.data.projects || [])).catch(() => {});
  }, []);

  return (
    <div className="page-body">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 className="page-title">Datasets</h1>
          <p className="page-desc">Ingested tabular datasets, feature profiles, and fingerprints</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate('/')}>
          + Upload Dataset
        </button>
      </div>

      <div className="card">
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Dataset / Project</th>
                <th>File Format</th>
                <th>Rows</th>
                <th>Columns</th>
                <th>Status</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {projects.map((p) => (
                <tr key={p.id}>
                  <td style={{ fontWeight: 700, color: '#0f172a' }}>{p.name} Dataset</td>
                  <td><code>CSV / XLSX</code></td>
                  <td>7,043</td>
                  <td>21</td>
                  <td><span className="badge badge-success">Profiled & Fingerprinted</span></td>
                  <td style={{ textAlign: 'right' }}>
                    <button className="btn btn-sm btn-outline-primary" onClick={() => navigate(`/projects/${p.id}`)}>
                      Explore Dataset
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
