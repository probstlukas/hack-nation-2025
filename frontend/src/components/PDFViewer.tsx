import React, { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCw, Download } from 'lucide-react';

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

interface PDFViewerProps {
  pdfUrl: string;
  documentName: string;
}

const PDFViewer: React.FC<PDFViewerProps> = ({ pdfUrl, documentName }) => {
  const [numPages, setNumPages] = useState<number>(0);
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [scale, setScale] = useState<number>(1.0);
  const [rotation, setRotation] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  // Fallback mode to native viewer
  const [mode, setMode] = useState<'react-pdf' | 'native'>('react-pdf');
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  // If react-pdf hasn't loaded within a few seconds, fallback to native
  useEffect(() => {
    setMode('react-pdf');
    setError(null);
    setNumPages(0);
    setPageNumber(1);

    const timer = window.setTimeout(() => {
      if (numPages === 0) {
        setMode('native');
      }
    }, 4000);

    return () => {
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pdfUrl]);

  // When switching to native mode, fetch the PDF as a blob and render via object URL
  useEffect(() => {
    let revoked = false;
    let currentUrl: string | null = null;

    async function loadBlob() {
      try {
        const res = await fetch(pdfUrl, { credentials: 'include' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        currentUrl = URL.createObjectURL(blob);
        if (!revoked) setBlobUrl(currentUrl);
      } catch (e) {
        setError('Failed to load PDF document');
      }
    }

    if (mode === 'native') {
      setBlobUrl(null);
      loadBlob();
    }

    return () => {
      revoked = true;
      if (currentUrl) URL.revokeObjectURL(currentUrl);
    };
  }, [mode, pdfUrl]);

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages);
    setError(null);
  }

  function onDocumentLoadError(err: Error) {
    console.error('PDF loading error:', err);
    // Switch to native fallback on error
    setMode('native');
  }

  const goToPrevPage = () => setPageNumber(page => Math.max(1, page - 1));
  const goToNextPage = () => setPageNumber(page => Math.min(numPages, page + 1));
  const zoomIn = () => setScale(s => Math.min(3.0, s + 0.2));
  const zoomOut = () => setScale(s => Math.max(0.5, s - 0.2));
  const rotate = () => setRotation(r => (r + 90) % 360);

  const downloadPDF = () => {
    const link = document.createElement('a');
    link.href = pdfUrl;
    link.download = `${documentName}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex flex-col h-full bg-gray-100">
      {/* PDF Toolbar */}
      <div className="bg-white border-b border-gray-300 p-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {mode === 'react-pdf' ? (
            <>
              <button onClick={goToPrevPage} disabled={pageNumber <= 1} className="btn btn-outline">
                <ChevronLeft size={16} />
              </button>
              <span className="text-sm text-gray-600">
                Page {numPages > 0 ? pageNumber : '-'} of {numPages > 0 ? numPages : '-'}
              </span>
              <button onClick={goToNextPage} disabled={pageNumber >= numPages || numPages === 0} className="btn btn-outline">
                <ChevronRight size={16} />
              </button>
            </>
          ) : (
            <span className="text-sm text-gray-600">Native PDF viewer</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {mode === 'react-pdf' && (
            <>
              <button onClick={zoomOut} disabled={scale <= 0.5} className="btn btn-outline" title="Zoom Out">
                <ZoomOut size={16} />
              </button>
              <span className="text-sm text-gray-600">{Math.round(scale * 100)}%</span>
              <button onClick={zoomIn} disabled={scale >= 3.0} className="btn btn-outline" title="Zoom In">
                <ZoomIn size={16} />
              </button>
              <button onClick={rotate} className="btn btn-outline" title="Rotate">
                <RotateCw size={16} />
              </button>
            </>
          )}

          <button onClick={downloadPDF} className="btn btn-primary" title="Download PDF">
            <Download size={16} />
            Download
          </button>
        </div>
      </div>

      {/* PDF Content */}
      <div className="flex-1 overflow-auto p-4 flex justify-center">
        <div className="pdf-container w-full h-full">
          {error && mode === 'native' && !blobUrl ? (
            <div className="flex items-center justify-center py-8 w-full h-full">
              <div className="text-center">
                <div className="text-red-600 mb-4">
                  <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <p className="text-gray-600 mb-4">{error}</p>
                <a href={pdfUrl} target="_blank" rel="noreferrer" className="btn btn-outline">Open in new tab</a>
              </div>
            </div>
          ) : mode === 'native' ? (
            blobUrl ? (
              <iframe
                title="PDF"
                src={blobUrl}
                className="w-full h-full"
                style={{ minHeight: '70vh', border: 0 }}
              />
            ) : (
              <div className="flex items-center justify-center py-8">
                <div className="spinner-primary"></div>
                <span className="ml-2">Loading PDF...</span>
              </div>
            )
          ) : (
            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={
                <div className="flex items-center justify-center py-8">
                  <div className="spinner-primary"></div>
                  <span className="ml-2">Loading document...</span>
                </div>
              }
            >
              <Page
                pageNumber={Math.min(pageNumber, Math.max(1, numPages))}
                scale={scale}
                rotate={rotation}
                loading={
                  <div className="flex items-center justify-center py-8">
                    <div className="spinner-primary"></div>
                    <span className="ml-2">Loading page...</span>
                  </div>
                }
              />
            </Document>
          )}
        </div>
      </div>

      {/* Page Input (Jump to page) */}
      {mode === 'react-pdf' && (
        <div className="bg-white border-t border-gray-300 p-3 flex items-center justify-center">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Go to page:</label>
            <input
              type="number"
              min={1}
              max={Math.max(1, numPages)}
              value={pageNumber}
              onChange={(e) => {
                const page = parseInt(e.target.value);
                if (!Number.isNaN(page)) setPageNumber(Math.min(Math.max(1, page), Math.max(1, numPages)));
              }}
              className="form-input w-20 text-center"
            />
            <span className="text-sm text-gray-600">of {numPages || '-'}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default PDFViewer;
