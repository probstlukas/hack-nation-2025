export const companyDomainOverrides: Record<string, string> = {
  '3M': '3m.com',
  'ACTIVISION BLIZZARD': 'activisionblizzard.com',
  'ADOBE': 'adobe.com',
  'AES': 'aes.com',
  'AMAZON': 'amazon.com',
  'AMCOR': 'amcor.com',
  'APPLE': 'apple.com',
  'MICROSOFT': 'microsoft.com',
  'GOOGLE': 'google.com',
  'ALPHABET': 'abc.xyz',
  'META': 'meta.com',
  'FACEBOOK': 'facebook.com',
  'NVIDIA': 'nvidia.com',
  'TESLA': 'tesla.com',
  'INTEL': 'intel.com',
  'IBM': 'ibm.com',
};

function normalizeName(name: string): string {
  return name.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim().toUpperCase();
}

function guessDomain(name: string): string {
  const cleaned = normalizeName(name)
    .replace(/[^A-Z0-9 ]/g, '') // keep alnum and spaces
    .replace(/\s+/g, '');
  return `${cleaned.toLowerCase()}.com`;
}

export function getCompanyLogoUrl(companyName: string): string {
  const key = normalizeName(companyName);
  const domain = companyDomainOverrides[key] || guessDomain(companyName);
  // Clearbit Logo API
  return `https://logo.clearbit.com/${domain}`;
}
