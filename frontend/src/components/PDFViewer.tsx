import React from 'react';

interface Props {
  pdfUrl: string;
  documentName?: string;
}

const PDFViewer: React.FC<Props> = ({ pdfUrl, documentName }) => {
  // Filter cross-origin "Script error." while this viewer is mounted (dev overlay noise)
  React.useEffect(() => {
    const onWindowError = (e: any) => {
      if (e && e.message === 'Script error.' && (!e.filename || e.filename === '')) {
        // Prevent dev overlay from showing a generic cross-origin error
        if (typeof e.preventDefault === 'function') e.preventDefault();
      }
    };
    window.addEventListener('error', onWindowError);
    return () => window.removeEventListener('error', onWindowError);
  }, []);

  return (
    <div style={{ height: '100%', minHeight: 600 }} onContextMenu={(e) => e.preventDefault()}>
      <object data={pdfUrl} type="application/pdf" width="100%" height="100%">
        <p>
          Unable to display the PDF. Download or open it here: {' '}
          <a href={pdfUrl} target="_blank" rel="noreferrer">
            {documentName || 'Open PDF'}
          </a>
        </p>
      </object>
    </div>
  );
};

export default PDFViewer;
