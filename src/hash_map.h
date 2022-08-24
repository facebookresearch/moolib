#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iterator>

#pragma push_macro("likely")
#pragma push_macro("unlikely")

#define likely(x) __builtin_expect(bool(x), 1)
#define unlikely(x) __builtin_expect(bool(x), 0)

#undef assert
#define assert(x) (bool(x) ? 0 : (printf("assert failure %s:%d\n", __FILE__, __LINE__), std::abort(), 0))

namespace moolib {

static constexpr int8_t none = -1;
template<typename T>
bool isNone(const T& v) {
  return v == (T)none;
}

template<typename Key, typename Value, typename Hash = std::hash<Key>>
struct HashMap {
private:
  struct PrimaryItem {
    Key key;
    Value value;
    size_t size;
  };
  struct SecondaryItem {
    Key key;
    Value value;
    size_t index;
  };

  size_t ksize = 0;
  size_t msize = 0;
  PrimaryItem* primary = nullptr;
  SecondaryItem* secondary = nullptr;

public:
  struct iterator {
  private:
  public:
    friend HashMap;
    HashMap* map;
    size_t ki;
    size_t vi;
    Value* v;
    mutable std::aligned_storage_t<sizeof(std::pair<Key&, Value&>), alignof(std::pair<Key&, Value&>)> tmp;

  public:
    iterator() = default;
    iterator(const HashMap* map, size_t ki, size_t vi, Value* v)
        : map(const_cast<HashMap*>(map)), ki(ki), vi(vi), v(v) {}
    iterator(const iterator& n) {
      map = n.map;
      ki = n.ki;
      vi = n.vi;
      v = n.v;
    }
    iterator& operator=(const iterator& n) {
      map = n.map;
      ki = n.ki;
      vi = n.vi;
      v = n.v;
      return *this;
    }

    using T = Value;

    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::bidirectional_iterator_tag;

    std::pair<Key&, Value&>& operator*() const noexcept {
      new (&tmp) std::pair<Key&, Value&>(*(Key*)((char*)v - sizeof(Key)), *v);
      return (std::pair<Key&, Value&>&)tmp;
    }
    std::pair<Key&, Value&>* operator->() const noexcept {
      return &**this;
    }

    iterator& operator++() noexcept {
      if (isNone(vi)) {
        do {
          if (ki == map->ksize - 1) {
            ki = none;
            vi = 0;
            v = &map->secondary[vi].value;
            if (!isNone(map->secondary[vi].index)) {
              return *this;
            }
            return ++*this;
          } else {
            ++ki;
            v = &map->primary[ki].value;
          }
        } while (isNone(map->primary[ki].size));
      } else {
        do {
          if (vi == map->ksize - 1) {
            v = nullptr;
            return *this;
          }
          ++vi;
          v = &map->secondary[vi].value;
        } while (isNone(map->secondary[vi].index));
      }
      return *this;
    }
    iterator operator++(int) noexcept {
      iterator r = *this;
      ++r;
      return r;
    }
    bool operator==(iterator n) const noexcept {
      return v == n.v;
    }
    bool operator!=(iterator n) const noexcept {
      return v != n.v;
    }
  };

  HashMap() = default;
  HashMap(const HashMap&) = delete;
  HashMap& operator=(const HashMap&) = delete;
  ~HashMap() {
    clear();
    deallocate(primary);
    deallocate(secondary);
  }

  bool empty() const noexcept {
    return msize == 0;
  }

  void clear() noexcept {
    auto i = begin();
    auto e = end();
    while (i != e) {
      i = erase(i);
    }
    assert(msize == 0);
    assert(begin() == end());
  }
  iterator begin() const noexcept {
    if (unlikely(!primary)) {
      return end();
    }
    size_t bs = ksize;
    for (size_t i = 0; i != bs; ++i) {
      if (!isNone(primary[i].size)) {
        return iterator(this, i, none, &primary[i].value);
      }
    }
    for (size_t i = 0; i != bs; ++i) {
      if (!isNone(secondary[i].index)) {
        return iterator(this, i, i, &secondary[i].value);
      }
    }
    return end();
  }
  iterator end() const noexcept {
    return iterator(this, 0, 0, nullptr);
  }

  template<typename T>
  T* allocate(size_t n) {
    T* retval = (T*)std::aligned_alloc(std::max(alignof(T), (size_t)32), std::max(sizeof(T) * n, (size_t)32));
    std::memset(retval, 0xcc, std::max(sizeof(T) * n, (size_t)32));
    return retval;
  }
  template<typename T>
  void deallocate(T* ptr) {
    if (ptr) {
      std::free(ptr);
    }
  }

  template<typename KeyT>
  void erase(KeyT&& key) {
    auto i = find(std::forward<KeyT>(key));
    if (i != end()) {
      erase(i);
    }
  }

  template<typename KeyT>
  Value& operator[](KeyT&& index) {
    return try_emplace(std::forward<KeyT>(index)).first->second;
  }

  iterator erase(iterator i) noexcept {
    auto mask = ksize - 1;
    size_t ki = i.ki;
    size_t vi = i.vi;
    assert(i.v);
    assert(ki < ksize);
    assert(isNone(i.vi) || vi < ksize);
    assert(msize > 0);
    --msize;
    auto* primary = this->primary;
    auto* secondary = this->secondary;
    if (likely(isNone(vi))) {
      auto& pv = primary[ki];
      size_t s = pv.size;
      if (unlikely(s)) {
        --s;
        size_t index = (ki + s) & mask;
        auto& sv = secondary[index];
        assert(sv.index == ki);
        sv.index = none;
        pv.key = std::move(sv.key);
        pv.value = std::move(sv.value);
        sv.key.~Key();
        sv.value.~Value();
        while (unlikely(s) && secondary[(ki + s - 1) & mask].index != ki) {
          --s;
        }
        pv.size = s;
        return i;
      }
      ++i;
      pv.size = none;
      pv.key.~Key();
      pv.value.~Value();
      assert(i == end() || (isNone(i.vi) ? !isNone(primary[i.ki].size) : !isNone(secondary[i.vi].index)));
      return i;
    } else {
      auto& sv = secondary[vi];
      if (isNone(ki)) {
        ki = sv.index;
      }
      auto& pv = primary[ki];
      size_t s = pv.size - 1;
      assert(pv.size > 0 && ((vi - ki) & mask) <= s);
      size_t lastIndex = (ki + s) & mask;
      auto& lv = secondary[lastIndex];
      assert(lv.index == ki);
      if (lastIndex == vi) {
        ++i;
      } else {
        sv.key = std::move(lv.key);
        sv.value = std::move(lv.value);
      }
      lv.index = none;
      lv.key.~Key();
      lv.value.~Value();
      while (s && secondary[(ki + s - 1) & mask].index != ki) {
        --s;
      }
      pv.size = s;
      assert(i == end() || (isNone(i.vi) ? !isNone(primary[i.ki].size) : !isNone(secondary[i.vi].index)));
      return i;
    }
  }

  template<typename KeyT, typename ValueT>
  iterator insert(KeyT&& key, ValueT&& value) {
    return try_emplace(std::forward<Key>(key), std::forward<Value>(value)).first;
  }

  void rehash(size_t newBs) noexcept {
    if (newBs & (newBs - 1)) {
      printf("bucket count is not a multiple of 2!\n");
      std::abort();
    }
    // printf("rehash %lu %lu\n", newBs, size());
    PrimaryItem* oldPrimary = primary;
    SecondaryItem* oldSecondary = secondary;

    primary = allocate<PrimaryItem>(newBs);
    secondary = allocate<SecondaryItem>(newBs);

    size_t bs = ksize;

    for (size_t i = 0; i != newBs; ++i) {
      primary[i].size = none;
      secondary[i].index = none;
    }

    size_t mmsize = msize;

    ksize = newBs;
    msize = 0;

    if (oldPrimary) {
      PrimaryItem* pend = oldPrimary + bs;
      for (auto* i = oldPrimary; i != pend; ++i) {
        if (!isNone(i->size)) {
          try_emplace(std::move(i->key), std::move(i->value));
          i->key.~Key();
          i->value.~Value();
        }
      }
      SecondaryItem* send = oldSecondary + bs;
      for (auto* i = oldSecondary; i != send; ++i) {
        if (!isNone(i->index)) {
          try_emplace(std::move(i->key), std::move(i->value));
          i->key.~Key();
          i->value.~Value();
        }
      }
    }

    deallocate(oldPrimary);
    deallocate(oldSecondary);
  }

  template<typename KeyT>
  iterator findSlowPath(size_t ki, KeyT&& key) const noexcept {
    size_t bs = ksize;
    size_t mask = bs - 1;
    auto* primary = this->primary;
    auto* secondary = this->secondary;
    auto& pv = primary[ki];
    auto equal = [](SecondaryItem* i, size_t ki, auto&& key) {
      if constexpr (std::is_trivial_v<KeyT> && std::is_trivial_v<Key>) {
        return (bool)(i->index == ki & i->key == key);
      }
      return i->index == ki && i->key == key;
    };

    size_t s = pv.size;
    size_t endvi = (ki + s) & mask;
    iterator enditerator(this, ki, endvi, nullptr);
    size_t vi = ki;
    SecondaryItem* b = secondary + vi;
    SecondaryItem* e = secondary + endvi;
    SecondaryItem* i = b;
    SecondaryItem* wrap = secondary + bs;
    do {
      if (likely(e > i)) {
        switch (s) {
        default:
        case 4:
          if (equal(i, ki, key)) {
            return iterator(this, ki, vi & mask, &i->value);
          }
          ++i;
          ++vi;
          --s;
        case 3:
          if (equal(i, ki, key)) {
            return iterator(this, ki, vi & mask, &i->value);
          }
          ++i;
          ++vi;
          --s;
        case 2:
          if (equal(i, ki, key)) {
            return iterator(this, ki, vi & mask, &i->value);
          }
          ++i;
          ++vi;
          --s;
        case 1:
          if (equal(i, ki, key)) {
            return iterator(this, ki, vi & mask, &i->value);
          }
          ++i;
          ++vi;
          --s;
          if (i == e) {
            return enditerator;
          }
        }
      }
      if (equal(i, ki, key)) {
        return iterator(this, ki, vi & mask, &i->value);
      }
      ++i;
      ++vi;
      --s;
      if (i == wrap) {
        i = secondary;
      }
    } while (i != e);
    return enditerator;
  }

  template<typename KeyT>
  iterator find(KeyT&& key) const noexcept {
    if (unlikely(!primary)) {
      return end();
    }
    size_t bs = ksize;
    size_t mask = bs - 1;
    size_t ki = Hash()(key) & mask;

    auto* primary = this->primary;
    auto* secondary = this->secondary;

    auto& pv = primary[ki];

    if (isNone(pv.size)) {
      return iterator(this, ki, none, nullptr);
    }
    if (likely(pv.key == key)) {
      return iterator(this, ki, none, &pv.value);
    }
    if (likely(pv.size == 0)) {
      return iterator(this, ki, ki, nullptr);
    }
    return findSlowPath(ki, std::forward<KeyT>(key));
  }

  void reserve(size_t n) {
    if (n >= std::numeric_limits<size_t>::max() / 2) {
      throw std::range_error("reserve beyond max size");
    }
    size_t bs = ksize;
    if (bs == 0) {
      bs = 1;
    }
    while (bs < n) {
      bs *= 2;
    }
    rehash(bs);
  }

  template<typename KeyT, typename... Args>
  std::pair<iterator, bool> try_emplace(KeyT&& key, Args&&... args) {
    if (unlikely(!primary)) {
      reserve(16);
    }
    auto i = find(key);
    if (unlikely(i.v)) {
      return std::make_pair(i, false);
    }
    size_t newksize = msize + 1;
    if (unlikely(ksize < newksize)) {
    resize:
      size_t omsize = msize;
      reserve(newksize);
      i = find(key);
      assert(i.v == nullptr);
      assert(msize == omsize);
    }
    auto* primary = this->primary;
    auto* secondary = this->secondary;
    auto& pv = primary[i.ki];
    if (likely(isNone(i.vi))) {
      ++msize;
      pv.size = 0;
      new (&pv.key) Key(std::forward<KeyT>(key));
      new (&pv.value) Value(std::forward<Args>(args)...);
      // printf("construct new primary value at %p\n", (void*)&pv.value);
      return std::make_pair(iterator(this, i.ki, none, &pv.value), true);
    } else if (ksize < 1024 * 1024 / sizeof(PrimaryItem) && ksize < msize * 4) {
    expand:
      newksize = ksize + 1;
      goto resize;
    } else {
      size_t mask = ksize - 1;
      size_t s = pv.size + 1;
      size_t index = i.vi;
      while (!isNone(secondary[index].index)) {
        index = (index + 1) & mask;
        ++s;
      }
      if ((unlikely(s >= 4) && ksize < 1024 * 1024 * 4 / sizeof(PrimaryItem)) || (unlikely(s >= ksize / 2))) {
        goto expand;
      }
      ++msize;
      pv.size = s;
      auto& sv = secondary[index];
      sv.index = i.ki;
      new (&sv.key) Key(std::forward<KeyT>(key));
      new (&sv.value) Value(std::forward<Args>(args)...);
      // printf("construct new secondary value at %p\n", (void*)&sv.value);
      return std::make_pair(iterator(this, i.ki, index, &sv.value), true);
    }
  }

  template<typename... Args>
  auto emplace(Args&&... args) {
    return try_emplace(std::forward<Args>(args)...);
  }

  size_t bucket_count() const noexcept {
    return ksize;
  }
  size_t size() const noexcept {
    return msize;
  }
};

} // namespace moolib

#pragma pop_macro("likely")
#pragma pop_macro("unlikely")
