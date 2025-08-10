import React from 'react';

interface TooltipProps {
  text: string;
  className?: string;
  children: React.ReactNode;
  placement?: 'top' | 'right' | 'bottom' | 'left';
}

const Tooltip: React.FC<TooltipProps> = ({ text, className, children, placement = 'top' }) => {
  const [visible, setVisible] = React.useState(false);

  const baseStyle: React.CSSProperties = {
    position: 'absolute',
    zIndex: 1000,
    background: '#111827', // gray-900
    color: '#fff',
    fontSize: 12,
    lineHeight: 1.3,
    padding: '6px 8px',
    borderRadius: 6,
    boxShadow: '0 8px 24px rgba(0,0,0,0.25)',
    whiteSpace: 'normal',
    maxWidth: 320,
    width: 'max-content',
    overflowWrap: 'break-word',
    wordBreak: 'break-word',
    pointerEvents: 'none',
    opacity: visible ? 1 : 0,
    transition: 'opacity 140ms ease, transform 140ms ease',
  };

  const posStyle: Record<string, React.CSSProperties> = {
    top: { bottom: '100%', left: '50%', transform: visible ? 'translate(-50%, -8px)' : 'translate(-50%, -2px)' },
    right: { left: '100%', top: '50%', transform: visible ? 'translate(8px, -50%)' : 'translate(2px, -50%)' },
    bottom: { top: '100%', left: '50%', transform: visible ? 'translate(-50%, 8px)' : 'translate(-50%, 2px)' },
    left: { right: '100%', top: '50%', transform: visible ? 'translate(-8px, -50%)' : 'translate(-2px, -50%)' },
  };

  return (
    <span
      className={className}
      style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      {children}
      <span role="tooltip" style={{ ...baseStyle, ...posStyle[placement] }}>{text}</span>
    </span>
  );
};

export default Tooltip;
