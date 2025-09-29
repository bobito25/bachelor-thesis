

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(ontable b)
(on c e)
(on d c)
(ontable e)
(ontable f)
(clear a)
(clear d)
(clear f)
)
(:goal
(and
(on a e)
(on b f)
(on c d)
(on e b))
)
)


