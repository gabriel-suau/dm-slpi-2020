// Ce bout de code nous sert pour contourner la limite de chiffres significatifs de std::to_string,
// qui nous sert pour les logs sur le terminal.
//
// Il a été entièrement copié du lien ci-dessous.
// https://stackoverflow.com/questions/16605967/set-precision-of-stdto-string-when-converting-floating-point-values
//

#ifndef STRING_ADD_ON_H
#define STRING_ADD_ON_H

#include <sstream>

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::scientific << a_value;
    return out.str();
}

#endif // STRING_ADD_ON_H
