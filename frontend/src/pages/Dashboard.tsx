import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  FileText, 
  Calendar, 
  Search, 
  BarChart3,
  ChevronRight
} from 'lucide-react';
import { Document } from '../types';
import { apiService } from '../services/api';
import { getCompanyLogoUrl } from '../utils/companyLogos';

const Dashboard: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [filteredDocuments, setFilteredDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCompany, setSelectedCompany] = useState('');
  const [selectedYear, setSelectedYear] = useState('');
  const [selectedType, setSelectedType] = useState('');

  useEffect(() => {
    loadDocuments();
  }, []);

  const filterDocuments = React.useCallback(() => {
    let filtered = documents;

    if (searchTerm) {
      filtered = filtered.filter(
        (doc) =>
          doc.company.toLowerCase().includes(searchTerm.toLowerCase()) ||
          doc.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (selectedCompany) {
      filtered = filtered.filter((doc) => doc.company === selectedCompany);
    }

    if (selectedYear) {
      filtered = filtered.filter((doc) => String(doc.year) === selectedYear || String(doc.period) === selectedYear);
    }

    if (selectedType) {
      filtered = filtered.filter((doc) => doc.doc_type.toLowerCase() === selectedType.toLowerCase());
    }

    setFilteredDocuments(filtered);
  }, [documents, searchTerm, selectedCompany, selectedYear, selectedType]);

  useEffect(() => {
    filterDocuments();
  }, [filterDocuments]);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      console.log('Loading documents from API...');
      const docs = await apiService.getDocuments();
      console.log('Documents loaded:', docs.length);
      setDocuments(docs);
      setError(null);
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to load documents';
      setError(`Failed to load documents: ${errorMsg}`);
      console.error('Error loading documents:', err);
    } finally {
      setLoading(false);
    }
  };

  const getUniqueCompanies = () => {
    return Array.from(new Set(documents.map((doc) => doc.company))).sort();
  };

  const getUniqueYears = () => {
    const pool = selectedCompany
      ? documents.filter((d) => d.company === selectedCompany)
      : documents;
    return Array.from(new Set(pool.map((doc) => String(doc.year || doc.period)))).sort();
  };

  const getUniqueTypes = () => {
    let pool = documents;
    if (selectedCompany) pool = pool.filter((d) => d.company === selectedCompany);
    if (selectedYear) pool = pool.filter((d) => String(d.year) === selectedYear || String(d.period) === selectedYear);
    return Array.from(new Set(pool.map((doc) => doc.doc_type.toUpperCase()))).sort();
  };

  const getDocTypeColor = (docType: string) => {
    switch (docType.toUpperCase()) {
      case '10K':
        return 'bg-blue-100 text-blue-800';
      case '10Q':
        return 'bg-green-100 text-green-800';
      case '8K':
        return 'bg-yellow-100 text-yellow-800';
      case 'EARNINGS':
        return 'bg-purple-100 text-purple-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto px-6 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="flex items-center gap-3">
            <div className="spinner-primary"></div>
            <span>Loading documents...</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white">
        <div className="container mx-auto px-6 py-16">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              FinDocGPT
            </h1>
            <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
              AI-powered financial document analysis and investment strategy platform
            </p>
            <div className="flex flex-col md:flex-row items-center justify-center gap-4">
              <div className="text-sm text-blue-100">
                Sponsored by <span className="font-semibold text-white">AkashX.ai</span>
              </div>
              <div className="text-sm text-blue-100">
                HackNation 2025 Challenge
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content: Document Library only (stages removed) */}
      <div className="container mx-auto px-6 py-12">
        {/* Document Library */}
        <div className="bg-white rounded-xl shadow-lg p-6 library-panel">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold">Financial Document Library</h2>
            <div className="text-sm text-gray-500">
              {filteredDocuments.length} of {documents.length} documents
            </div>
          </div>

          {/* Search and Cascading Filters */}
          <div className="grid md:grid-cols-4 gap-4 mb-6">
            <div className="md:col-span-1 col-span-2">
              <div className="input-group">
                <input
                  type="text"
                  placeholder="Search by company or document name..."
                  className="form-input"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
                <Search className="input-icon" size={18} />
              </div>
            </div>

            {/* Company */}
            <div className="select-wrap">
              <select
                className="form-input"
                value={selectedCompany}
                onChange={(e) => {
                  setSelectedCompany(e.target.value);
                  setSelectedYear('');
                  setSelectedType('');
                }}
              >
                <option value="">All Companies</option>
                {getUniqueCompanies().map((company) => (
                  <option key={company} value={company}>
                    {company}
                  </option>
                ))}
              </select>
            </div>

            {/* Year */}
            <div className="select-wrap">
              <select
                className="form-input"
                value={selectedYear}
                onChange={(e) => {
                  setSelectedYear(e.target.value);
                  setSelectedType('');
                }}
                disabled={!selectedCompany && getUniqueYears().length === 0}
              >
                <option value="">All Years</option>
                {getUniqueYears().map((year) => (
                  <option key={year} value={year}>
                    {year}
                  </option>
                ))}
              </select>
            </div>

            {/* Type */}
            <div className="select-wrap">
              <select
                className="form-input"
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                disabled={getUniqueTypes().length === 0}
              >
                <option value="">All Types</option>
                {getUniqueTypes().map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Error State */}
          {error && (
            <div className="alert alert-danger mb-6">
              {error}
              <button onClick={loadDocuments} className="btn btn-outline ml-4">
                Retry
              </button>
            </div>
          )}

          {/* Documents Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredDocuments.map((doc) => (
              <Link
                key={doc.id}
                to={`/document/${doc.id}`}
                className="card doc-card hover:shadow-lg transition-all duration-200 group"
              >
                <div className="card-body">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      {/* Company Logo */}
                      <img
                        src={getCompanyLogoUrl(doc.company)}
                        alt={`${doc.company} logo`}
                        className="company-logo"
                        width={28}
                        height={28}
                        loading="lazy"
                        decoding="async"
                        onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
                      />
                      <span className="font-semibold text-gray-900">{doc.company}</span>
                    </div>
                    <ChevronRight 
                      size={16} 
                      className="text-gray-400 group-hover:text-blue-600 transition-colors" 
                    />
                  </div>

                  <div className="mb-3">
                    <span className={`badge ${getDocTypeColor(doc.doc_type)}`}>
                      {doc.doc_type}
                    </span>
                  </div>

                  <div className="space-y-2 text-sm text-gray-600">
                    <div className="flex items-center gap-2">
                      <Calendar size={14} />
                      <span>Period: {doc.period}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <BarChart3 size={14} />
                      <span>Sector: {doc.sector}</span>
                    </div>
                  </div>

                  {doc.financial_metrics && (
                    <div className="mt-3 pt-3 border-t border-gray-100">
                      <div className="text-xs text-gray-500">Financial Data Available</div>
                    </div>
                  )}
                </div>
              </Link>
            ))}
          </div>

          {filteredDocuments.length === 0 && !loading && (
            <div className="text-center py-12">
              <FileText size={48} className="text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No documents found matching your criteria.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
